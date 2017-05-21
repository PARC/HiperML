/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

/*
 * hml_sgemv_kernel.cu:
 * GEneral Matrix-Matrix multiplication (GEMV) between two matrices
 * A and B, where A is M rows x K columns and B is K rows x N columns.
 * Unlike CUBLAS and MAGMA, which use column-major store, this program assumes
 * both A and B are stored in row-major order.
 */
#include "hml_sgemv_kernel.h"
#include <vector>
#include <algorithm>
#include <limits>
#include "hml_flops.h"

#define HML_SGEMV_CONFIG_HEADER \
  "#SGEMV smart kernel config file v1.0\n"\
  "#SGEMV: Single-precision general matrix-vector multiplication\n"\
  "#formula: y = alpha * A * x + beta * y\n"\
  "#where:   A is M x N\n"\
  "#         x is N x 1\n"\
  "#         y is M x 1\n"\
  "#         alpha and beta are scalars\n"\
  "#format: N : kernelType blockStops rowStops\n"\
  "#where:\n"\
  "#  N          : number of columns of A\n"\
  "#  kernelType : {basic = 0, var-K = 1, const-K = 2}\n"\
  "#  blockStops : (# threads per thread block) / 16\n" \
  "#  rowStops   : (#rows of A assigned to a thread block) / (blockStops*16)\n"\
  "#example config rule:\n"\
  "#1 : 2 3 4\n"\
  "#example meaning:\n"\
  "#  if number of columns of A is 1, then choose the\n"\
  "#  const-K SGEMV kernel with 3 block stops and 4 row stops\n"\

/* global variable declaration */
HmlSgemvKernelRepo   sgemvKernelRepo;
bool              hmlSgemvKernelInitialized = false;
HmlSgemvKernelConfig hmlSgemvKernelConfig[cHmlMaxSkinnyN+1];
bool              hmlSgemvKernelConfigDone = false;

#ifdef HML_USE_TEXTURE_MEM
/* external global variable declaration */
extern texture<float, 1> texFloatA;
extern texture<float, 1> texFloatx;
#endif /* HML_USE_TEXTURE_MEM */

/* function declarations */
void
hmlSgemvKernelRepoInit(HmlSgemvKernelVarN  *basic,
                    HmlSgemvKernelVarN   varN[][cHmlMaxStops+1],
                    HmlSgemvKernelConstN constN[][cHmlMaxBlockStops+1][cHmlMaxStops+1]);

void hmlSgemvKernelArgSet(HmlKernelArg       *arg,
                       int              M,
                       int              N,
                       HmlSgemvKernelType  type,
                       int              blockStops,
                       int              rowStops)
{
  int AsubRows;
  int rowBlocks;

  arg->block.x = blockStops * 16;
  arg->block.y = 1;
  arg->block.z = 1;
  if (type != eHmlSgemvKernelBasic)
    AsubRows = rowStops * blockStops * 16;
  else
    AsubRows = 1;
  rowBlocks = (M + AsubRows - 1) / AsubRows;
  arg->grid.x = min(rowBlocks, cHmlMaxGridDimX);
  arg->grid.y = (rowBlocks + cHmlMaxGridDimX - 1) / cHmlMaxGridDimX;
  arg->grid.z = 1;
  arg->allocBytes = 0;
  /*
    fprintf(stderr, "block.x = %d, block.y = %d, block.z = %d\n",
            arg->block.x, arg->block.y, arg->block.z);
    fprintf(stderr, "grid.x = %d, grid.y = %d\n", arg->grid.x, arg->grid.y);
  */
}


/* Single-precision general matrix-vector multiplication (sGEMV) kernel.
 * Matrix A is not transposed, and thus 'N' function name suffix
 * we assume row-major order.
 */
__global__ void
hmlSgemvKernel1T1RGmemN(float       *y,
               const float *A,
               const float *x,
               const int    M,
               const int    N,
               const float  alpha,
               const float  beta)
{
  const int row = blockIdx.x * 128 + threadIdx.x;
  float ax = 0.0;
  A += row * N;

  for (int i = 0; i < (N >> 5); ++i, A += 32, x += 32) {
#pragma unroll
    for (int j = 0; j < 32; ++j)
      ax += A[j] * x[j];
  }
  for (int j = 0; j < (N & 31); ++j)
    ax += A[j] * x[j];

  if (row < M)
    y[row] = alpha * ax + beta * y[row];
}

/* Single-precision general matrix-vector multiplication (sGEMV) kernel.
 * Matrix A is not transposed, and thus 'N' function name suffix
 * we assume row-major order.
 */
__global__ void
hmlSgemvKernel1T1RSmemN(float       *y,
               const float *A,
               const float *x,
               const int    M,
               const int    N,
               const float  alpha,
               const float  beta)
{
  const int tid = threadIdx.x;
  const int row = blockIdx.x * 128 + tid;

  float ax = 0.0;
  __shared__ float xs[128];
  A += row * N;

  for (int i = 0; i < (N >> 5); ++i, A += 128, x += 128) {
    __syncthreads();
    xs[tid] = x[tid];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 128; ++j)
      ax += A[j] * xs[j];
  }
  __syncthreads();
  if (N & 31) {
    xs[tid] = x[tid];
    __syncthreads();
    for (int j = 0; j < (N & 31); ++j)
      ax += A[j] * xs[j];
  }

  if (row < M)
    y[row] = alpha * ax + beta * y[row];
}

/* Single-precision general matrix-vector multiplication (sGEMV) kernel.
 * Matrix A is not transposed, and thus 'N' function name suffix
 * we assume row-major order.
 * 128T1R: 128 threads per row
 * shared memory is used to perform the 128 -> 1 reduction
 */
__global__ void
hmlSgemvKernel128T1RN(float       *y,
             const float *A,
             const float *x,
             const int    M,
             const int    N,
             const float  alpha,
             const float  beta)
{
  /* assuming one-dimensional thread block */
  const int tid = threadIdx.x;
  __shared__ float yVal[128];
  /* needed for reduction without __syncthreads() */
  volatile float *yValInWarp;
  float ax = 0.0f;
  /* row is set to be the row # in A[] for this thread block */
  const int row = blockIdx.x + blockIdx.y * gridDim.x;
  //const int baseIdxA = row * N;
  /* N16 is the number of A columns rounded down to the nearest
   * multiple of 128
   */
  const int N16 = N / 128 * 128;

  /* each thread computes 1/128-th sub-vector of y */
  if (row < M) {
    /* loop over all the sub-matrices of A and
     * the sub-vectors of x in stride 16
     * multiply each <sub-matrix, sub-vector> pair together
     * and accumulate the results
     */
    int n;
    for (n = 0; n < N16; n += 128) {
      ax += A[tid + n + N * blockIdx.x] * x[tid + n];
    }
    if (tid + N16 < N)
      ax += A[tid + N16 + N * blockIdx.x] * x[tid + N16];
    else
      ax += 0.0f;
    yVal[tid] = ax;
    /* synchronize before reduction */
    __syncthreads();
    if (tid < 64)
      yVal[tid] += yVal[tid + 64];
    __syncthreads();
    if (tid < 32) {
      yValInWarp = yVal;
      yValInWarp[tid] += yValInWarp[tid + 32];
      yValInWarp[tid] += yValInWarp[tid + 16];
      yValInWarp[tid] += yValInWarp[tid + 8];
      yValInWarp[tid] += yValInWarp[tid + 4];
      yValInWarp[tid] += yValInWarp[tid + 2];
      yValInWarp[tid] += yValInWarp[tid + 1];
    }
    /* write y[row] to device memory */
    if (tid == 0)
      y[row] = alpha * yVal[0] + beta * y[row];
  }
}

void
hmlSgemvKernelReset(int             N,
                 HmlSgemvKernelType type,
                 int             blockStops,
                 int             rowStops)

{
  if (type == eHmlSgemvKernelBasic)
    sgemvKernelRepo.basic[blockStops] = NULL;
  else if (type == eHmlSgemvKernelVarN)
    sgemvKernelRepo.varN[blockStops][rowStops] = NULL;
  else if (type == eHmlSgemvKernelConstN)
    sgemvKernelRepo.constN[N][blockStops][rowStops] = NULL;
  else {
    fprintf(stderr, "; Error: Invalid kernel type '%d'\n", type);
    exit(1);
  }
}

void
hmlSgemvKernelGet(HmlSgemvKernelVarN   *varN,
               HmlSgemvKernelConstN *constN,
               int                N,
               HmlSgemvKernelType    type,
               int                blockStops,
               int                rowStops)
{
  if (type == eHmlSgemvKernelBasic) {
    if (varN)
      *varN   = sgemvKernelRepo.basic[blockStops];
    if (constN)
      *constN = NULL;
  }
  else if (type == eHmlSgemvKernelVarN) {
    if (varN)
      *varN   = sgemvKernelRepo.varN[blockStops][rowStops];
    if (constN)
      *constN = NULL;
  }
  else if (type == eHmlSgemvKernelConstN) {
    if (varN)
      *varN   = NULL;
    if (constN)
      *constN = sgemvKernelRepo.constN[N][blockStops][rowStops];
  }
  else {
    if (varN)
      *varN   = NULL;
    if (constN)
      *constN = NULL;
    fprintf(stderr, "; Error: Invalid kernel type '%d'\n", type);
    exit(1);
  }
}

void
hmlSgemvKernelSelectBasic(HmlSgemvKernelVarN   *basic,
                       HmlKernelArg         *karg,
                       const int          M,
                       const int          N,
                       int                blockStops)
{
  if (basic) {
    *basic = sgemvKernelRepo.basic[blockStops];
    if (*basic == NULL) {
      fprintf(stderr, "; Error: missing SGEMV basic kernel %d", blockStops);
      exit(1);
    }
  }
  if (karg)
    hmlSgemvKernelArgSet(karg, M, N, eHmlSgemvKernelBasic, blockStops, 1);
}

/* uses global array: hmlSgemvKernelConfig[] to pick the best
 * tall-and-skinny SGEMV kernel
 */
void
hmlSgemvKernelSelect(HmlSgemvKernelVarN   *varN,
                  HmlSgemvKernelConstN *constN,
                  HmlKernelArg         *karg,
                  const int          M,
                  const int          N)
{
  HmlSgemvKernelType type;
  int             blockStops;
  int             rowStops;

  if (N <= cHmlMaxSkinnyN) {
    type       = hmlSgemvKernelConfig[N].type;
    blockStops = hmlSgemvKernelConfig[N].blockStops;
    rowStops   = hmlSgemvKernelConfig[N].rowStops;
  }
  else { /* N > cHmlMaxSkinnyN */
    type       = hmlSgemvKernelConfig[0].type;
    blockStops = hmlSgemvKernelConfig[0].blockStops;
    rowStops   = hmlSgemvKernelConfig[0].rowStops;
  }
  /* is this an invalid config rule? */
  if (blockStops <= 0 || rowStops <= 0) {
    type = eHmlSgemvKernelBasic; /* use basic kernel for invalid rules */
    blockStops = cHmlSgemvKernelBasicBlockStops;
    rowStops   = cHmlSgemvKernelBasicStops;
  }
  hmlSgemvKernelGet(varN, constN, N, type, blockStops, rowStops);
  /* even if the rule is valid, let's check if the desired kernel
   * is actually available or not. If not, fall back to basic kernel
   */
  if (*constN == NULL && *varN == NULL) {
    type = eHmlSgemvKernelBasic;
    blockStops = cHmlSgemvKernelBasicBlockStops;
    rowStops   = cHmlSgemvKernelBasicStops;
    hmlSgemvKernelGet(varN, constN, N, type, blockStops, rowStops);
  }
  fprintf(stderr, "Info: kernel type = %d, blockStops = %d, rowStops = %d\n",
          type, blockStops, rowStops);
  /* setup kernel parameters */
  hmlSgemvKernelArgSet(karg, M, N, type, blockStops, rowStops);
}

void hmlSgemvKernelInit(void)
{
  if (!hmlSgemvKernelInitialized) {
    hmlSgemvKernelRepoInit(sgemvKernelRepo.basic,
                        sgemvKernelRepo.varN,
                        sgemvKernelRepo.constN);
    if (sgemvKernelRepo.basic[cHmlSgemvKernelBasicBlockStops] == NULL) {
      fprintf(stderr, "; Error: basic SGEMV kernel missing\n");
      exit(1);
    }
    hmlSgemvKernelInitialized = true;
  }
}

void
hmlSgemvKernelSetOpt(
  int    n,
  int    optBlockStops[cHmlSgemvKernelTypes],
  int    optRowStops[cHmlSgemvKernelTypes],
  double GFLOPS[cHmlMaxSkinnyN+1][cHmlSgemvKernelTypes][cHmlMaxBlockStops+1][cHmlMaxStops+1])
{
  int    type;
  int    optType = 0;
  double maxGFLOPS = 0.0;

  for (type = 0; type < cHmlSgemvKernelTypes; ++type) {
    if (GFLOPS[n][type][optBlockStops[type]][optRowStops[type]] > maxGFLOPS) {
      maxGFLOPS = GFLOPS[n][type][optBlockStops[type]][optRowStops[type]];
      optType = type;
    }
  }
  hmlSgemvKernelConfig[n].type       = (HmlSgemvKernelType)optType;
  hmlSgemvKernelConfig[n].blockStops = optBlockStops[optType];
  hmlSgemvKernelConfig[n].rowStops   = optRowStops[optType];
}

void
hmlSgemvKernelConfigOnline(int maxMxN, int testTrialsPerKernel, int verbosity)
{
  double cpuStart, cpuEnd;
  double wallStart, wallEnd;
  std::vector<double> trialGFLOPS;
  double GFLOPS[cHmlMaxSkinnyN+1][cHmlSgemvKernelTypes][cHmlMaxBlockStops+1][cHmlMaxStops+1];
  int numTestScenarios = cHmlSgemvKernelTypes *
    (cHmlMaxSkinnyN + 1) * (cHmlMaxBlockStops + 1) * (cHmlMaxStops + 1);
  float *hostA, *hostx, *hosty; /* y = A * x */
  int M, N;     /* A is M x N, x is N x 1, y is M x 1 */
  int blockStops, rowStops;
  double gflop;   /* giga floating point operation */
  double minGFLOPS;
  double maxGFLOPS;
  int    optBlockStops[cHmlSgemvKernelTypes];
  int    optRowStops[cHmlSgemvKernelTypes];
  float *devA, *devx, *devy; /* arrays on CUDA device */
  HmlSgemvKernelConstN constN = NULL;
  HmlSgemvKernelVarN   varN   = NULL;
  HmlKernelArg         karg;
  int               type;
  cudaError_t       err;
  int               testFailures = 0;

  if (!hmlSgemvKernelInitialized)
    hmlSgemvKernelInit();
  hostA = (float*)malloc(maxMxN * sizeof(float));
  hostx = (float*)malloc(cHmlMaxTestN * sizeof(float));
  hosty = (float*)malloc(cHmlMaxTestM * sizeof(float));
  if (!hostA || !hostx || !hosty) {
    fprintf(stderr, "; Error: out of main memory\n");
    exit(1);
  }
  /* initialize hostA, hostx, and hosty */
  memset(hostA, 11, maxMxN * sizeof(float));
  memset(hostx, 22, cHmlMaxTestN * sizeof(float));
  memset(hosty, 33, cHmlMaxTestM * sizeof(float));
  /* alloc and load A, x, and y to device memory */
#ifdef HML_USE_TEXTURE_MEM
  devA = hmlDeviceFloatArrayAllocLoadBind(hostA, maxMxN, texFloatA);
  devx = hmlDeviceFloatArrayAllocLoadBind(hostx, cHmlMaxTestN, texFloatx);
#else
  devA = hmlDeviceFloatArrayAllocLoad(hostA, maxMxN);
  devx = hmlDeviceFloatArrayAllocLoad(hostx, cHmlMaxTestN);
#endif /* HML_USE_TEXTURE_MEM */
  devy = hmlDeviceFloatArrayAllocLoad(hosty, cHmlMaxTestM);
  /* init GFLOPS 4D array */
  memset(GFLOPS, 0, sizeof(double) * numTestScenarios);
  memset(hmlSgemvKernelConfig, 0, sizeof(HmlSgemvKernelConfig) * (cHmlMaxSkinnyN + 1));
  /* loop over values of N */
  for (int n = 0; n <= cHmlMaxSkinnyN; ++n) {
    N = (n) ? n : cHmlMaxTestN;
    M = min(maxMxN / N, cHmlMaxTestM);
    /* round to the nearest 1000s to avoid a problem that
     * causes kernels to crash on Nvidia K2000 cards
     */
    M = M / 1000 * 1000;
    gflop = FLOPS_SGEMV(M, N) / 1e9;
    for (type = 0; type < cHmlSgemvKernelTypes; ++type) {
      minGFLOPS = std::numeric_limits<double>::max();
      maxGFLOPS = 0.0;
      optBlockStops[type] = 0;
      optRowStops[type] = 0;
      for (blockStops = 1; blockStops <= cHmlMaxBlockStops; ++blockStops) {
          for (rowStops = 1; rowStops <= cHmlMaxStops; ++rowStops) {
          if (type == eHmlSgemvKernelBasic) {
            varN = sgemvKernelRepo.basic[blockStops];
            /* skip any kernel that is not instantiated */
            if (!varN)
              break; /* exit rowStop loop now */
          }
          else if (type == eHmlSgemvKernelVarN) {
            varN = sgemvKernelRepo.varN[blockStops][rowStops];
            /* skip any kernel that is not instantiated */
            if (!varN)
              continue;
          }
          else if (type == eHmlSgemvKernelConstN) {
            if (N <= cHmlMaxSkinnyN)
              constN = sgemvKernelRepo.constN[N][blockStops][rowStops];
            else
              constN = NULL;
            /* skip any kernel that is not instantiated */
            if (!constN)
              continue;
          }
          else
            continue;
          hmlSgemvKernelArgSet(&karg, M, N, (HmlSgemvKernelType)type,
                            blockStops, rowStops);
          trialGFLOPS.clear();
          /* avoid warning that err may be uninitialized */
          err = cudaSuccess;
          for (int trial = 0; trial < testTrialsPerKernel; ++trial) {
            hmlGetSecs(&cpuStart, &wallStart);   /* start the timer */
            /* invoke sGEMV kernal to compute y = alpha * A * x + beta * y */
            if (type == eHmlSgemvKernelBasic || type == eHmlSgemvKernelVarN) {
              varN<<<karg.grid, karg.block, karg.allocBytes>>>(
                devy, devA, devx, M, N, 1.0, 0.0);
            }
            else if (type == eHmlSgemvKernelConstN) {
              constN<<<karg.grid, karg.block, karg.allocBytes>>>(
                devy, devA, devx, M, 1.0, 0.0);
            }
            else
              continue;
            /* block until kernel completed */
            err = cudaDeviceSynchronize();
            hmlGetSecs(&cpuEnd, &wallEnd); /* stop the timer */
            if (err != cudaSuccess) {
              fprintf(stderr,
                      "Warning: kernel[%2d][%1d][%2d][%2d] test failed '%s'\n",
                      n, type, blockStops, rowStops, cudaGetErrorString(err));
              hmlSgemvKernelReset(N, (HmlSgemvKernelType)type, blockStops, rowStops);
              ++testFailures;
              break;
            }
            trialGFLOPS.push_back(gflop / (wallEnd - wallStart));
            if (verbosity >= 3) {
              fprintf(stderr,
                      "Info: kernel[%2d][%1d][%2d][%2d] = %7.2lf GFLOPS\n",
                      n, type, blockStops, rowStops,
                      gflop / (wallEnd - wallStart));
            }
          }
          if (err == cudaSuccess) {
            std::sort(trialGFLOPS.begin(), trialGFLOPS.end());
            /* store the median GFLOPS */
            GFLOPS[n][type][blockStops][rowStops] =
              trialGFLOPS[testTrialsPerKernel / 2];
            if (GFLOPS[n][type][blockStops][rowStops] > maxGFLOPS) {
              maxGFLOPS = GFLOPS[n][type][blockStops][rowStops];
              optBlockStops[type] = blockStops;
              optRowStops[type]   = rowStops;
            }
            if (verbosity >= 2) {
              fprintf(stderr,
                      "Info: kernel[%2d][%1d][%2d][%2d] = %7.2lf GFLOPS\n",
                      n, type, blockStops, rowStops,
                      GFLOPS[n][type][blockStops][rowStops]);
            }
            minGFLOPS = min(minGFLOPS, GFLOPS[n][type][blockStops][rowStops]);
          }
          if (type == eHmlSgemvKernelBasic)
            break; /* basic kernel doesn't have rowStops > 1 */
        }
      }
      if (verbosity >= 1 &&
          GFLOPS[n][type][optBlockStops[type]][optRowStops[type]] > 0.0) {
        fprintf(stderr,
                "Info: kernel*[%2d][%1d][%2d][%2d] = %7.2lf GFLOPS, "
                "max / min = %3.0lf%%\n",
                n, type, optBlockStops[type], optRowStops[type],
                GFLOPS[n][type][optBlockStops[type]][optRowStops[type]],
                maxGFLOPS * 100 / minGFLOPS);
        fflush(stderr);
      }
    }
    if (verbosity >= 2) {
      fprintf(stderr, "varN / base = %3.0lf%%, constN / base = %3.0lf%%\n",
              GFLOPS[n][eHmlSgemvKernelVarN][optBlockStops[eHmlSgemvKernelVarN]][optRowStops[eHmlSgemvKernelVarN]] * 100 /
              GFLOPS[n][eHmlSgemvKernelBasic][optBlockStops[eHmlSgemvKernelBasic]][optRowStops[eHmlSgemvKernelBasic]],
              GFLOPS[n][eHmlSgemvKernelConstN][optBlockStops[eHmlSgemvKernelConstN]][optRowStops[eHmlSgemvKernelConstN]] * 100 /
              GFLOPS[n][eHmlSgemvKernelBasic][optBlockStops[eHmlSgemvKernelBasic]][optRowStops[eHmlSgemvKernelBasic]]);
    }
    hmlSgemvKernelSetOpt(n, optBlockStops, optRowStops, GFLOPS);
  }
  /* depending on the use case, this may be OK in the future, since
   * all the failed kernels were reset (i.e., invalidated) by
   * hmlSgemvKernelReset(). But for now, we stop the program if
   * there is any test failures
   */
  if (testFailures > 0) {
    fprintf(stderr, "; Error: Number of failed tests = %d\n", testFailures);
    exit(1);
  }

  /* output GFLOPS summary */
  fprintf(stderr, "GFLOPS summary:\n");
  fprintf(stderr, "   N   Basic    VarN  ConstN\n");
  fprintf(stderr, "============================\n");
  for (int n = 0; n <= cHmlMaxSkinnyN; ++n) {
    fprintf(stderr, "%4d", n);
    for (type = 0; type < cHmlSgemvKernelTypes; ++type) {
      maxGFLOPS = 0.0;
      for (blockStops = 1; blockStops <= cHmlMaxBlockStops; ++blockStops)
        for (rowStops = 1; rowStops <= cHmlMaxStops; ++rowStops)
          maxGFLOPS = max(maxGFLOPS, GFLOPS[n][type][blockStops][rowStops]);
      fprintf(stderr, " %7.2lf", maxGFLOPS);
    }
    fprintf(stderr, "\n");
  }
  /* output max / min summary */
  fprintf(stderr, "max / min summary:\n");
  fprintf(stderr, "   N   Basic    VarN  ConstN\n");
  fprintf(stderr, "============================\n");
  for (int n = 0; n <= cHmlMaxSkinnyN; ++n) {
    fprintf(stderr, "%4d", n);
    for (type = 0; type < cHmlSgemvKernelTypes; ++type) {
      minGFLOPS = std::numeric_limits<double>::max();
      maxGFLOPS = 0.0;
      for (blockStops = 1; blockStops <= cHmlMaxBlockStops; ++blockStops) {
        for (rowStops = 1; rowStops <= cHmlMaxStops; ++rowStops) {
          if (GFLOPS[n][type][blockStops][rowStops] > 0.0)
            minGFLOPS = min(minGFLOPS, GFLOPS[n][type][blockStops][rowStops]);
          maxGFLOPS = max(maxGFLOPS, GFLOPS[n][type][blockStops][rowStops]);
        }
      }
      fprintf(stderr, "    %3.0lf%%", maxGFLOPS * 100 / minGFLOPS);
    }
    fprintf(stderr, "\n");
  }

  hmlSgemvKernelConfigDone = true;

#ifdef HML_USE_TEXTURE_MEM
  /* unbind texture memory */
  cudaUnbindTexture(texFloatA);
  cudaUnbindTexture(texFloatx);
#endif /* HML_USE_TEXTURE_MEM */

  /* free device memory */
  cudaFree(devA);
  cudaFree(devx);
  cudaFree(devy);

  /* free host memory */
  free(hostA);
  free(hostx);
  free(hosty);
}

void
hmlSgemvKernelConfigPrint(FILE *file)
{
  fprintf(file, HML_SGEMV_CONFIG_HEADER);
  for (int N = 0; N <= cHmlMaxSkinnyN; ++N) {
    if (hmlSgemvKernelConfig[N].rowStops > 0) {
      fprintf(file, "%d : %d %d %d\n", N,
              hmlSgemvKernelConfig[N].type,
              hmlSgemvKernelConfig[N].blockStops,
              hmlSgemvKernelConfig[N].rowStops);
    }
  }
}

/* read kernel config file into global array:
 * hmlSgemvKernelConfig[][]
 */
void
hmlSgemvKernelConfigReadFile(const char *fileName)
{
  FILE       *file;
  char        line[cHmlLineBufferSize];
  char       *str;
  int         N;
  int         type;
  int         blockStops;
  int         rowStops;

  memset(hmlSgemvKernelConfig, 0, sizeof(HmlSgemvKernelConfig) * (cHmlMaxSkinnyN + 1));
  file = openFile(fileName, "rb");
  while (!feof(file)) {
    while (true) {
      str = fgets(line, cHmlLineBufferSize, file);
      if (!str || *str != '#')
        break;
    }
    if (!str) break;
    sscanf(str, "%d : %d %d %d\n", &N, &type, &blockStops, &rowStops);
    hmlSgemvKernelConfig[N].type       = (HmlSgemvKernelType)type;
    hmlSgemvKernelConfig[N].blockStops = blockStops;
    hmlSgemvKernelConfig[N].rowStops   = rowStops;
  }
  fclose(file);
  hmlSgemvKernelConfigDone = true;
}
