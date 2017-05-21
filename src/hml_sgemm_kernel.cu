/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/
/*
 * hml_sgemm_kernel.cu:
 * GEneral Matrix-Matrix multiplication (GEMM) between two matrices
 * A and B, where A is M rows x K columns and B is K rows x N columns.
 * Unlike CUBLAS and MAGMA, which use column-major store, this program assumes
 * both A and B are stored in row-major order.
 */
#include "hml_sgemm_kernel.h"
#include "hml_flops.h"
#include <vector>
#include <algorithm>
#include <limits>

#define HML_SGEMM_CONFIG_HEADER \
  "#SGEMM smart kernel config file v1.0\n"\
  "#SGEMM: Single-precision general matrix-matrix multiplication\n"\
  "#formula: C = alpha * A * B + beta * C\n"\
  "#where:   A is M x K\n"\
  "#         B is K x N\n"\
  "#         C is M x N\n"\
  "#         alpha and beta are scalars\n"\
  "#format: K colStops : kernelType rowStops\n"\
  "#where:\n"\
  "#  K          : number of columns of A\n"\
  "#  colStops   : (N + 15) / 16, number of column stops\n"\
  "#  kernelType : {basic = 0, var-K = 1, const-K = 2}\n"\
  "#  rowStops   : (#rows of A assigned to a 16x16 thread block) / 16\n"\
  "#example config rule:\n"\
  "#2 1 : 2 16\n"\
  "#example meaning:\n"\
  "#  if number of columns of A is 2 and number of column stops is 1,\n"\
  "#  then choose the const-K SGEMM kernel with 16 row stops\n"\
 
#ifdef HML_USE_TEXTURE_MEM
extern texture<float, 1> texFloatA;
extern texture<float, 1> texFloatB;
#endif /* HML_USE_TEXTURE_MEM */

/* global variable declaration */
HmlSgemmKernelRepo   sgemmKernelRepo;
bool              sgemmKernelRepoInitialized = false;
HmlSgemmKernelConfig hmlSgemmKernelConfig[cHmlMaxSkinnyK+1][cHmlMaxStops+1];
bool              sgemmKernelConfigDone = false;

/* function declarations */
void
hmlSgemmKernelInit(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]);

void
initSGemmKernel1(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]);

void
initSGemmKernel2(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]);

void
initSGemmKernel3(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]);

void
initSGemmKernel4(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]);

void
initSGemmKernel5(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]);

void
initSGemmKernel6(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]);

void hmlSgemmKernelArgSet(HmlKernelArg *arg,
                          int        M,
                          int        N,
                          int        K,
                          int        colStops,
                          int        rowStops) {
  arg->block.x = 64;
  arg->block.y = 4;
  arg->block.z = 1;
  int AsubRows = 16 * rowStops;
  int rowBlocks = (M + AsubRows - 1) / AsubRows;
  arg->grid.x = min(rowBlocks, cHmlMaxGridDimX);
  arg->grid.y = (rowBlocks + cHmlMaxGridDimX - 1) / cHmlMaxGridDimX;
  int BsubCols = 16 * colStops;
  int colBlocks = (N + BsubCols - 1) / BsubCols;
  if(colBlocks > cHmlMaxGridDimZ) {
    fprintf(stderr, "; Error: # of column blocks exceeds system limit (%d)\n",
            cHmlMaxGridDimZ);
    exit(1);
  }
  arg->grid.z = colBlocks;
  arg->allocBytes = 0;
}

void
hmlSgemmKernelReset(int             K,
                    HmlSgemmKernelType type,
                    int             colStops,
                    int             rowStops) {
  if(type == eHmlSgemmKernelBasic) {
    sgemmKernelRepo.basic = NULL;
  }
  else if(type == eHmlSgemmKernelVarK) {
    sgemmKernelRepo.varK[colStops][rowStops] = NULL;
  }
  else if(type == eHmlSgemmKernelConstK) {
    sgemmKernelRepo.constK[K][colStops][rowStops] = NULL;
  }
  else {
    fprintf(stderr, "; Error: Invalid kernel type '%d'\n", type);
    exit(1);
  }
}

void
hmlSgemmKernelGet(HmlSgemmKernelVarK   *varK,
                  HmlSgemmKernelConstK *constK,
                  int                K,
                  HmlSgemmKernelType    type,
                  int                colStops,
                  int                rowStops) {
  if(type == eHmlSgemmKernelBasic) {
    if(varK) {
      *varK   = sgemmKernelRepo.basic;
    }
    if(constK) {
      *constK = NULL;
    }
  }
  else if(type == eHmlSgemmKernelVarK) {
    if(varK) {
      *varK   = sgemmKernelRepo.varK[colStops][rowStops];
    }
    if(constK) {
      *constK = NULL;
    }
  }
  else if(type == eHmlSgemmKernelConstK) {
    if(varK) {
      *varK   = NULL;
    }
    if(constK) {
      *constK = sgemmKernelRepo.constK[K][colStops][rowStops];
    }
  }
  else {
    if(varK) {
      *varK   = NULL;
    }
    if(constK) {
      *constK = NULL;
    }
    fprintf(stderr, "; Error: Invalid kernel type '%d'\n", type);
    exit(1);
  }
}

void
hmlSgemmKernelSelectBasic(HmlSgemmKernelVarK   *basic,
                          HmlKernelArg         *karg,
                          const int          M,
                          const int          N,
                          const int          K) {
  if(basic)
    *basic =
      sgemmKernelRepo.varK[cHmlSgemmKernelBasicStops][cHmlSgemmKernelBasicStops];
  if(karg)
    hmlSgemmKernelArgSet(karg, M, N, K,
                         cHmlSgemmKernelBasicStops, cHmlSgemmKernelBasicStops);
}

/* uses global array: hmlSgemmKernelConfig[][] to pick the best
 * tall-and-skinny SGEMM kernel
 */
void
hmlSgemmKernelSelect(HmlSgemmKernelVarK   *varK,
                     HmlSgemmKernelConstK *constK,
                     HmlKernelArg         *karg,
                     const int          M,
                     const int          N,
                     const int          K) {
  HmlSgemmKernelType type;
  int             colStops;
  int             rowStops;

  if(N <= cHmlMaxSkinnyN) {
    colStops = (N + 15) / 16;
  }
  else { /* N > cHmlMaxSkinnyN */
    colStops = (cHmlMaxSkinnyN + 15) / 16;
  }

  if(K <= cHmlMaxSkinnyK) {
    type     = hmlSgemmKernelConfig[K][colStops].type;
    rowStops = hmlSgemmKernelConfig[K][colStops].rowStops;
  }
  else { /* K > cHmlMaxSkinnyK */
    type     = hmlSgemmKernelConfig[cHmlMaxSkinnyK][colStops].type;
    rowStops = hmlSgemmKernelConfig[cHmlMaxSkinnyK][colStops].rowStops;
  }
  /* is this an invalid config rule? */
  if(rowStops <= 0) {
    type = eHmlSgemmKernelBasic; /* use basic kernel for invalid rules */
    rowStops = colStops = cHmlSgemmKernelBasicStops;
  }
  hmlSgemmKernelGet(varK, constK, K, type, colStops, rowStops);
  /* even if the rule is valid, let's check if the desired kernel
   * is actually available or not. If not, fall back to basic kernel
   */
  if(*constK || *varK) {
    /* kernel available, let's setup kernel arguments */
    hmlSgemmKernelArgSet(karg, M, N, K, colStops, rowStops);
  }
  else {
    /* kernel not found, fal back to the basic */
    type = eHmlSgemmKernelBasic;
    rowStops = colStops = cHmlSgemmKernelBasicStops;
    /* kernel arguments are also set by hmlSgemmKernelSelectBasic */
    hmlSgemmKernelSelectBasic(varK, karg, M, N, K);
  }
  fprintf(stderr, "Info: kernel type = %d, colStops = %d, rowStops = %d\n",
          type, colStops, rowStops);
}

/* Some compute capabilities (e.g., 2.0) would terminate
 * kernels that use too many registers per thread, causing
 * the on-line config routine to believe the kernel runs
 * extremely "fast." Calling this function ensures the
 * integrity of the SGEMM kernel repository.
 */
void
hmlSgemmKernelResetInvalid(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]) {
  int colStops;
  int rowStops;
  int K;
  cudaDeviceProp prop;

  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  for(colStops = 1; colStops <= cHmlMaxStops; ++colStops) {
    for(rowStops = 1; rowStops <= cHmlMaxStops; ++rowStops) {
      /* the following inequality test is only a heuristic! */
      if(colStops * rowStops > hmlMaxNumRegistersPerThread(&prop)) {
        varK[colStops][rowStops] = NULL;
        for(K = 1; K <= cHmlMaxSkinnyK; ++K) {
          constK[K][colStops][rowStops] = NULL;
        }
      }
    }
  }
}

void
hmlSgemmKernelInit(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]) {
  int numConstKKernels = (cHmlMaxSkinnyK + 1) * (cHmlMaxStops + 1) * (cHmlMaxStops + 1);
  int numVarKKernels = (cHmlMaxStops + 1) * (cHmlMaxStops + 1);
  memset(constK, 0, sizeof(HmlSgemmKernelConstK) * numConstKKernels);
  memset(varK, 0, sizeof(HmlSgemmKernelVarK) * numVarKKernels);

  initSGemmKernel1(varK, constK);
  initSGemmKernel2(varK, constK);
  initSGemmKernel3(varK, constK);
  initSGemmKernel4(varK, constK);
  initSGemmKernel5(varK, constK);
  initSGemmKernel6(varK, constK);
  hmlSgemmKernelResetInvalid(varK, constK);
}

void hmlSgemmKernelRepoInit(void) {
  if(!sgemmKernelRepoInitialized) {
    hmlSgemmKernelInit(sgemmKernelRepo.varK, sgemmKernelRepo.constK);
    hmlSgemmKernelSelectBasic(&sgemmKernelRepo.basic, NULL, 0, 0, 0);
    if(!sgemmKernelRepo.basic) {
      fprintf(stderr, "; Error: SGEMM basic kernel not found.\n");
      exit(1);
    }
    sgemmKernelRepoInitialized = true;
  }
}

void
hmlSgemmKernelSetOpt(
  int    K,
  int    colStops,
  int    optRowStops[cHmlSgemmKernelTypes],
  double GFLOPS[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlSgemmKernelTypes][cHmlMaxStops+1]) {
  int    type;
  int    optType = 0;
  double maxGFLOPS = 0.0;

  for(type = 0; type < cHmlSgemmKernelTypes; ++type) {
    if(GFLOPS[K][colStops][type][optRowStops[type]] > maxGFLOPS) {
      maxGFLOPS = GFLOPS[K][colStops][type][optRowStops[type]];
      optType = type;
    }
  }
  hmlSgemmKernelConfig[K][colStops].type = (HmlSgemmKernelType)optType;
  hmlSgemmKernelConfig[K][colStops].rowStops = optRowStops[optType];
}

void
hmlSgemmKernelConfigPrintStats(double GFLOPS[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlSgemmKernelTypes][cHmlMaxStops+1],
                               int type, char const *typeStr) {
  /* output GFLOPS summary */
  fprintf(stderr, "%s GFLOPS:\n", typeStr);
  fprintf(stderr, "   K    N=16    N=32    N=48    N=64    N=80    N=96\n");
  fprintf(stderr, "====================================================\n");
  for(int K = 2; K <= cHmlMaxSkinnyK; ++K) {
    fprintf(stderr, "%4d", K);
    for(int colStops = 1; colStops <= (cHmlMaxSkinnyN + 15) / 16; ++colStops) {
      double maxGFLOPS = 0.0;
      for(int rowStops = 1; rowStops <= cHmlMaxStops; ++rowStops) {
        maxGFLOPS = max(maxGFLOPS, GFLOPS[K][colStops][type][rowStops]);
      }
      fprintf(stderr, " %7.2lf", maxGFLOPS);
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "%s max / min:\n", typeStr);
  fprintf(stderr, "   K    N=16    N=32    N=48    N=64    N=80    N=96\n");
  fprintf(stderr, "====================================================\n");
  for(int K = 2; K <= cHmlMaxSkinnyK; ++K) {
    fprintf(stderr, "%4d", K);
    for(int colStops = 1; colStops <= (cHmlMaxSkinnyN + 15) / 16; ++colStops) {
      double minGFLOPS = std::numeric_limits<double>::max();
      double maxGFLOPS = 0.0;
      for(int rowStops = 1; rowStops <= cHmlMaxStops; ++rowStops) {
        if(GFLOPS[K][colStops][type][rowStops] > 0.0) {
          minGFLOPS = min(minGFLOPS, GFLOPS[K][colStops][type][rowStops]);
          maxGFLOPS = max(maxGFLOPS, GFLOPS[K][colStops][type][rowStops]);
        }
      }
      fprintf(stderr, "    %3.0lf%%", maxGFLOPS * 100 / minGFLOPS);
    }
    fprintf(stderr, "\n");
  }
}

void
hmlSgemmKernelConfigOnline(int testRows, int testTrialsPerKernel, int verbosity) {
  double cpuStart, cpuEnd;
  double wallStart, wallEnd;
  std::vector<double> trialGFLOPS;
  double GFLOPS[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlSgemmKernelTypes][cHmlMaxStops+1];
  int numTestScenarios = cHmlSgemmKernelTypes *
                         (cHmlMaxSkinnyK + 1) * (cHmlMaxStops + 1) * (cHmlMaxStops + 1);
  float *hostA, *hostB, *hostC; /* C = A x B */
  int M, N, K;     /* A is M x K, B is K x N, C is M x N */
  int maxN;
  int colStops, rowStops;
  double gflop;   /* giga floating point operation */
  double minGFLOPS;
  double maxGFLOPS;
  int    optRowStops[cHmlSgemmKernelTypes];
  float *devA, *devB, *devC; /* arrays on CUDA device */
  HmlSgemmKernelConstK constK = NULL;
  HmlSgemmKernelVarK   varK   = NULL;
  HmlKernelArg         karg;
  int               type;
  cudaError_t       err;
  int               testFailures = 0;

  if(!sgemmKernelRepoInitialized) {
    hmlSgemmKernelRepoInit();
  }
  M = testRows;
  maxN = cHmlMaxStops * 16;
  hostA = (float *)malloc(M * cHmlMaxSkinnyK * sizeof(float));
  hostB = (float *)malloc(maxN * cHmlMaxSkinnyK * sizeof(float));
  hostC = (float *)malloc(M * maxN * sizeof(float));
  if(!hostA || !hostB || !hostC) {
    fprintf(stderr, "; Error: out of main memory\n");
    exit(1);
  }
  /* initialize hostA, hostx, and hosty */
  memset(hostA, 11, M * cHmlMaxSkinnyK * sizeof(float));
  memset(hostB, 22, maxN * cHmlMaxSkinnyK * sizeof(float));
  memset(hostC, 33, M * maxN * sizeof(float));
  /* alloc and load A, B, and C to device memory */
#ifdef HML_USE_TEXTURE_MEM
  devA = hmlDeviceFloatArrayAllocLoadBind(hostA, M * cHmlMaxSkinnyK, texFloatA);
  devB = hmlDeviceFloatArrayAllocLoadBind(hostB, maxN * cHmlMaxSkinnyK, texFloatB);
#else
  devA = hmlDeviceFloatArrayAllocLoad(hostA, M * cHmlMaxSkinnyK);
  devB = hmlDeviceFloatArrayAllocLoad(hostB, maxN * cHmlMaxSkinnyK);
#endif /* HML_USE_TEXTURE_MEM */
  devC = hmlDeviceFloatArrayAllocLoad(hostC, M * maxN);
  /* init GFLOPS 4D array */
  memset(GFLOPS, 0, sizeof(double) * numTestScenarios);
  memset(hmlSgemmKernelConfig, 0, sizeof(HmlSgemmKernelConfig) *
         (cHmlMaxSkinnyK + 1) * (cHmlMaxStops + 1));
  /* loop over values of K */
  for(K = 2; K <= cHmlMaxSkinnyK; ++K) {
    for(colStops = 1; colStops <= (cHmlMaxSkinnyN + 15) / 16; ++colStops) {
      N = colStops * 16;
      gflop = FLOPS_SGEMM(M, N, K) / 1e9;
      for(type = 0; type < cHmlSgemmKernelTypes; ++type) {
        minGFLOPS = std::numeric_limits<double>::max();
        maxGFLOPS = 0.0;
        optRowStops[type] = 0;
        rowStops = (type == eHmlSgemmKernelBasic) ? 6 : 1;
        for(; rowStops <= cHmlMaxStops; ++rowStops) {
          if(type == eHmlSgemmKernelBasic) {
            varK = sgemmKernelRepo.basic;
            /* skip any kernel that is not instantiated */
            if(!varK) {
              continue;
            }
          }
          else if(type == eHmlSgemmKernelVarK) {
            varK = sgemmKernelRepo.varK[colStops][rowStops];
            /* skip any kernel that is not instantiated */
            if(!varK) {
              continue;
            }
          }
          else if(type == eHmlSgemmKernelConstK) {
            constK = sgemmKernelRepo.constK[K][colStops][rowStops];
            /* skip any kernel that is not instantiated */
            if(!constK) {
              continue;
            }
          }
          else {
            continue;
          }
          hmlSgemmKernelArgSet(&karg, M, N, K, colStops, rowStops);
          trialGFLOPS.clear();
          /* avoid warning that err may be uninitialized */
          err = cudaSuccess;
          for(int trial = 0; trial < testTrialsPerKernel; ++trial) {
            hmlGetSecs(&cpuStart, &wallStart);   /* start the timer */
            /* invoke sGEMM kernal to compute C = alpha x A x B + beta x C */
            if(type == eHmlSgemmKernelBasic || type == eHmlSgemmKernelVarK) {
              varK<<<karg.grid, karg.block, karg.allocBytes>>>(
                devC, devA, devB, M, N, K, 1.0, 0.0);
            }
            else if(type == eHmlSgemmKernelConstK) {
              constK<<<karg.grid, karg.block, karg.allocBytes>>>(
                devC, devA, devB, M, N, 1.0, 0.0);
            }
            else {
              continue;
            }
            /* block until kernel completed */
            err = cudaDeviceSynchronize();
            hmlGetSecs(&cpuEnd, &wallEnd); /* stop the timer */
            if(err != cudaSuccess) {
              fprintf(stderr,
                      "Warning: kernel[%2d][%2d][%1d][%2d] test failed '%s'\n",
                      K, colStops, type, rowStops, cudaGetErrorString(err));
              hmlSgemmKernelReset(K, (HmlSgemmKernelType)type, colStops, rowStops);
              ++testFailures;
              break;
            }
            trialGFLOPS.push_back(gflop / (wallEnd - wallStart));
            if(verbosity >= 3) {
              fprintf(stderr,
                      "Info: kernel[%2d][%2d][%1d][%2d] = %7.2lf GFLOPS\n",
                      K, colStops, type, rowStops,
                      gflop / (wallEnd - wallStart));
            }
          }
          if(err == cudaSuccess) {
            std::sort(trialGFLOPS.begin(), trialGFLOPS.end());
            GFLOPS[K][colStops][type][rowStops] =
              trialGFLOPS[testTrialsPerKernel / 2];
            if(GFLOPS[K][colStops][type][rowStops] > maxGFLOPS) {
              maxGFLOPS = GFLOPS[K][colStops][type][rowStops];
              optRowStops[type] = rowStops;
            }
            if(verbosity >= 2) {
              fprintf(stderr,
                      "Info: kernel[%2d][%2d][%1d][%2d] = %7.2lf GFLOPS\n",
                      K, colStops, type, rowStops,
                      GFLOPS[K][colStops][type][rowStops]);
            }
            minGFLOPS = min(minGFLOPS, GFLOPS[K][colStops][type][rowStops]);
          }
          if(type == eHmlSgemmKernelBasic) {
            break;  /* basic kernel only has a fixed number of rowStops */
          }
        }
        if(verbosity >= 1 &&
            GFLOPS[K][colStops][type][optRowStops[type]] > 0.0) {
          fprintf(stderr,
                  "Info: kernel*[%2d][%2d][%1d][%2d] = %7.2lf GFLOPS",
                  K, colStops, type, optRowStops[type],
                  GFLOPS[K][colStops][type][optRowStops[type]]);
          if(type != eHmlSgemmKernelBasic) {
            fprintf(stderr, ", max / min = %3.0lf%%", maxGFLOPS*100/minGFLOPS);
          }
          fprintf(stderr, "\n");
          fflush(stderr);
        }
      }
      if(verbosity >= 2) {
        fprintf(stderr, "varK / gen = %3.0lf%%, constK / gen = %3.0lf%%\n",
                GFLOPS[K][colStops][eHmlSgemmKernelVarK][optRowStops[eHmlSgemmKernelVarK]] * 100 /
                GFLOPS[K][colStops][eHmlSgemmKernelBasic][optRowStops[eHmlSgemmKernelBasic]],
                GFLOPS[K][colStops][eHmlSgemmKernelConstK][optRowStops[eHmlSgemmKernelConstK]] * 100 /
                GFLOPS[K][colStops][eHmlSgemmKernelBasic][optRowStops[eHmlSgemmKernelBasic]]);
      }
      hmlSgemmKernelSetOpt(K, colStops, optRowStops, GFLOPS);
    }
  }
  /* depending on the use case, this may be OK in the future, since
   * all the failed kernels were reset (i.e., invalidated) by
   * hmlSgemmKernelReset(). But for now, we stop the program if
   * there is any test failures
   */
  if(testFailures > 0) {
    fprintf(stderr, "; Error: Number of failed tests = %d\n", testFailures);
    exit(1);
  }
  /* print config stats */
  hmlSgemmKernelConfigPrintStats(GFLOPS, eHmlSgemmKernelBasic, "Basic");
  hmlSgemmKernelConfigPrintStats(GFLOPS, eHmlSgemmKernelVarK, "VarK");
  hmlSgemmKernelConfigPrintStats(GFLOPS, eHmlSgemmKernelConstK, "ConstK");

  sgemmKernelConfigDone = true;

#ifdef HML_USE_TEXTURE_MEM
  /* unbind texture memory */
  cudaUnbindTexture(texFloatA);
  cudaUnbindTexture(texFloatB);
#endif /* HML_USE_TEXTURE_MEM */

  /* free device memory */
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  /* free host memory */
  free(hostA);
  free(hostB);
  free(hostC);
}

void
hmlSgemmKernelConfigPrint(FILE *file) {
  fprintf(file, HML_SGEMM_CONFIG_HEADER);
  for(int K = 2; K <= cHmlMaxSkinnyK; ++K) {
    for(int colStops = 1; colStops <= cHmlMaxStops; ++colStops) {
      if(hmlSgemmKernelConfig[K][colStops].rowStops > 0)
        fprintf(file, "%d %d : %d %d\n", K, colStops,
                hmlSgemmKernelConfig[K][colStops].type,
                hmlSgemmKernelConfig[K][colStops].rowStops);
    }
  }
}

/* read kernel config file into global array:
 * hmlSgemmKernelConfig[][]
 */
void
hmlSgemmKernelConfigReadFile(const char *fileName) {
  FILE       *file;
  char        line[cHmlLineBufferSize];
  char       *str;
  int         K;
  int         colStops;
  int         rowStops;
  int         type;

  memset(hmlSgemmKernelConfig, 0, sizeof(HmlSgemmKernelConfig) *
         (cHmlMaxSkinnyK + 1) * (cHmlMaxStops + 1));
  file = openFile(fileName, "rb");
  while(!feof(file)) {
    while(true) {
      str = fgets(line, cHmlLineBufferSize, file);
      if(!str || *str != '#') {
        break;
      }
    }
    if(!str) {
      break;
    }
    sscanf(str, "%d %d : %d %d\n", &K, &colStops, &type, &rowStops);
    hmlSgemmKernelConfig[K][colStops].type     = (HmlSgemmKernelType)type;
    hmlSgemmKernelConfig[K][colStops].rowStops = rowStops;
  }
  fclose(file);
  sgemmKernelConfigDone = true;
}
