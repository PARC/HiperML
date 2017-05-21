/*
 * hml_sgemv_test.cu: GEneral Matrix-Matrix multiplication (GEMV)
 * between a matrix A and a vector x, where A is M rows x N columns
 * and x is N columns. 
 * Unlike CUBLAS and MAGMA, which use column-major store, this program assumes
 * A is stored in row-major order.
 */
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <limits>
#ifdef WIN32
#include "getopt_win32.h"
#else
#include <unistd.h>
#endif

#include "hml_utils.h"
#include "hml_consts.h"
#include "hml_types.h"
#include "hml_utils.h"
#include "hml_flops.h"
#include "hml_sgemv.h"
#include "hml_sgemv_kernel.h"

#define cHmlSgemvRelativeErrorMax         (1.0e-8)

#ifdef WIN32
#define cHmlSgemvConfigTestMxNMax        20000000
#define cHmlSgemvConfigTestsPerKernel    3
#else
#define cHmlSgemvConfigTestMxNMax        100000000
#define cHmlSgemvConfigTestsPerKernel    5
#endif /* WIN32 */

#ifdef HML_USE_TEXTURE_MEM
/* external global variable declaration */
extern texture<float, 1> texFloatA;
extern texture<float, 1> texFloatx;
#endif /* HML_USE_TEXTURE_MEM */

/* Basic, non-smart API function for single precision GEMV (sGEMV)
 * for computing C = alpha * A * B + beta * C on the GPU
 * Note: all matrices should not have any storage paddings. 
 */
void
hmlSgemvBasicTest(float       *y,
	   const float *A,
	   const float *x,
	   int          M,
	   int          N,
	   float        alpha,
	   float        beta,
	   int          trials,
	   double      *medianGFLOPS)
{
  double cpuStart, wallStart;   /* time keeping variables */
  double cpuEnd, wallEnd;   /* time keeping variables */
  /* compute the GFLOP based on sizes of the matrices */
  double gflop =      FLOPS_SGEMV(M, N) / 1e9;
  double              GFLOPS;
  std::vector<double> trialGFLOPS;
  float *devA, *devx, *devy; /* matrices on CUDA device */
  cudaError_t         err;
  HmlSgemvKernelVarN     sgemvBasicKernel;
  HmlKernelArg           karg;
  
#ifdef HML_USE_TEXTURE_MEM
  /* cHmlMaxCudaTexture1DLinear is in terms of elements instead of bytes */
  if (M * N >= cHmlMaxCudaTexture1DLinear) {
    fprintf(stderr, "CUDA maxTexture1DLinear exceeded\n");
    exit(1);
  }
#endif /* HML_USE_TEXTURE_MEM */
  
  /* alloc float array on device, load from host, and possibly
   * bind the device array to texture memory for A & x 
   */
#ifdef HML_USE_TEXTURE_MEM
  devA = hmlDeviceFloatArrayAllocLoadBind(A, M * N, texFloatA);
  devx = hmlDeviceFloatArrayAllocLoadBind(x, N, texFloatx);
#else
  devA = hmlDeviceFloatArrayAllocLoad(A, M * N);
  devx = hmlDeviceFloatArrayAllocLoad(x, N);
#endif /* HML_USE_TEXTURE_MEM */
  
  /* alloc and load float array on device for vector y */
  devy = hmlDeviceFloatArrayAllocLoad(y, M);
  
  /* setup kernel arguments */
  hmlSgemvKernelSelectBasic(&sgemvBasicKernel, &karg, M, N,
			 cHmlSgemvKernelBasicBlockStops);
  //sgemvBasicKernel = hmlSgemvKernel128T1RN; //blockStops must == 8
  for (int t = 0; t < trials; ++t) {
    hmlGetSecs(&cpuStart, &wallStart);   /* start the timer */
    /* invoke sGEMV basic kernal to compute y = alpha * A * x + beta * y */
    sgemvBasicKernel<<<karg.grid, karg.block>>>(devy,
						devA,
						devx,
						M,
						N,
						alpha,
						beta);
    /*
      if (M <= 8500)
      hmlSgemvKernel1T1RGmemN<<<(M+127)/128, 128>>>(devy, devA, devx, M, N, alpha, beta);
      else
      hmlSgemvKernel1T1RSmemN<<<(M+127)/128, 128>>>(devy, devA, devx, M, N, alpha, beta);
    */
  
    /* block until computation is completed */
    err = cudaDeviceSynchronize(); 
    hmlGetSecs(&cpuEnd, &wallEnd); /* stop the timer */
    if (err != cudaSuccess) {
      fprintf(stderr, "Run kernel: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    GFLOPS = gflop / (wallEnd - wallStart);
    //fprintf(stderr, "Trial#%d: GFLOPS = %.2lf\n", t, GFLOPS);
    trialGFLOPS.push_back(GFLOPS);
  }
  std::sort(trialGFLOPS.begin(), trialGFLOPS.end());
  *medianGFLOPS = trialGFLOPS[trials / 2];
  //fprintf(stderr, "Median GFLOPS = %.2lf\n", *medianGFLOPS);
  /* copy y from device memory to host memory */
  err = cudaMemcpy(y, devy, M * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "; Error: copy y from device: %s\n",cudaGetErrorString(err));
    exit(1);
  }

#ifdef HML_USE_TEXTURE_MEM
  /* unbind texture memory */
  cudaUnbindTexture(texFloatA);
  cudaUnbindTexture(texFloatx);
#endif /* HML_USE_TEXTURE_MEM */

  /* free device memory */
  cudaFree(devA);
  cudaFree(devx);
  cudaFree(devy);
}

/* The "smart" version of the API function for single precision GEMV
 * (sGEMV) for computing y = alpha * A * x + beta * y,
 * where it is the best if matrix A is tall and skinny.
 * Note: all matrices and vectors should not have any storage paddings.
 */
void 
hmlSgemvTest(float       *y,
	  const float *A,
	  const float *x,
	  int          M,
	  int          N,
	  float        alpha,
	  float        beta,
	  int          trials,
	  double      *medianGFLOPS) 
{
  double cpuStart, wallStart;   /* time keeping variables */
  double cpuEnd, wallEnd;   /* time keeping variables */
  /* compute the GFLOPS based on sizes of the matrices */
  double gflop =      FLOPS_SGEMV(M, N) / 1e9;
  double              GFLOPS;
  std::vector<double> trialGFLOPS;
  float *devA, *devx, *devy; /* matrices on CUDA device */
  cudaError_t         err;
  HmlSgemvKernelConstN   constN;
  HmlSgemvKernelVarN     varN;
  HmlKernelArg           karg;
  
#ifdef HML_USE_TEXTURE_MEM
  /* cHmlMaxCudaTexture1DLinear is in terms of elements instead of bytes */
  if (M * N >= cHmlMaxCudaTexture1DLinear) {
    fprintf(stderr, "CUDA maxTexture1DLinear exceeded\n");
    exit(1);
  }
#endif /* HML_USE_TEXTURE_MEM */
  
  /* alloc float array on device, load from host, and possibly
   * bind the device array to texture memory for A & x 
   */
#ifdef HML_USE_TEXTURE_MEM
  devA = hmlDeviceFloatArrayAllocLoadBind(A, M * N, texFloatA);
  devx = hmlDeviceFloatArrayAllocLoadBind(x, N, texFloatx);
#else
  devA = hmlDeviceFloatArrayAllocLoad(A, M * N);
  devx = hmlDeviceFloatArrayAllocLoad(x, N);
#endif /* HML_USE_TEXTURE_MEM */

  /* alloc and load float array on device for result vector y */
  devy = hmlDeviceFloatArrayAllocLoad(y, M);

  /* pick the right kernel to run */
  hmlSgemvKernelSelect(&varN, &constN, &karg, M, N);
  for (int t = 0; t < trials; ++t) {
    hmlGetSecs(&cpuStart, &wallStart);   /* start the timer */
    /* invoke sGEMV kernal to compute y = alpha * A * x + beta * y */
    if (constN) {
      constN<<<karg.grid, karg.block>>>(devy,
					devA,
					devx,
					M,
					alpha,
					beta);
    }
    else {
      varN<<<karg.grid, karg.block>>>(devy,
				      devA,
				      devx,
				      M,
				      N,
				      alpha,
				      beta);
    }
    /* block until computation is completed */
    err = cudaDeviceSynchronize(); 
    hmlGetSecs(&cpuEnd, &wallEnd); /* stop the timer */
    if (err != cudaSuccess) {
      fprintf(stderr, "Run kernel: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    GFLOPS = gflop / (wallEnd - wallStart);
    //fprintf(stderr, "Trial#%d: GFLOPS = %.2lf\n", t, GFLOPS);
    trialGFLOPS.push_back(GFLOPS);
  }
  std::sort(trialGFLOPS.begin(), trialGFLOPS.end());
  *medianGFLOPS = trialGFLOPS[trials / 2];
  //fprintf(stderr, "Median GFLOPS = %.2lf\n", *medianGFLOPS);
  /* copy vector y from device memory to host memory */
  err = cudaMemcpy(y, devy, M * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "; Error: copy y from device: %s\n",cudaGetErrorString(err));
    exit(1);
  }

#ifdef HML_USE_TEXTURE_MEM
  /* unbind texture memory */
  cudaUnbindTexture(texFloatA);
  cudaUnbindTexture(texFloatx);
#endif /* HML_USE_TEXTURE_MEM */
  
  /* free device memory */
  cudaFree(devA);
  cudaFree(devx);
  cudaFree(devy);
}

/* test function for single precision GEMV (sGEMV) on CPU
 * NOTE: the dimension of matrix or vector does NOT have to be multiples of 16
 * for best performance, it is desirable to have M and N as multiples of 32.
 * we assume row-major order.
 */
void
hmlSgemvCpu(float       *y,
         const float *A,
         const float *x,
         const int    M,
         const int    N,
         const float  alpha,
         const float  beta)
{
  double cpuStart, wallStart;   /* time keeping variables */
  double cpuEnd, wallEnd;   /* time keeping variables */
  /* compute the giga flop based on sizes of the matrices */
  double gflop = FLOPS_SGEMV(M, N) / 1e9;    
  const float *Aend = A + M * N;
  float  ax;
  
  hmlGetSecs(&cpuStart, &wallStart);   /* start the timer */
  while (A < Aend) {
    ax = 0.0;
    for (int j = 0; j < N; ++j)
      ax += A[j] * x[j];
    *y = alpha * ax + beta * *y;
    ++y;
    A += N;
  }
  hmlGetSecs(&cpuEnd, &wallEnd); /* stop the timer */
  fprintf(stderr, "CPU time = %lf, wall time = %lf\n",
          cpuEnd - cpuStart, wallEnd - wallStart);
  fprintf(stderr, "GFLOPS = %lf\n", gflop / (wallEnd - wallStart));  
}

double hmlComputeYsum(float *y, int M)
{
  double ysum = 0.0;
  for(int i = 0; i < M; i++) {
    ysum += y[i];
  }
  return ysum;
}

int main(int argc, char* argv[]) 
{
  char                       helpMsg[] =
    "This program does GPU-based Single-precision "
    "GEneral Matrix-Vector multiplication (SGEMV):\n"
    "y = alpha * A * x + beta * y\n"
    "where A is M x N, x is N x 1, and y is M x 1\n"
    "Options:\n"
    "\t-c Cross check with CPU-based SGEMM results\n"
    "\t-h Print this help message\n"
    "\t-M <M>\n"
    "\t-N <N>\n"
    "\t-o <output smart-kernel config file name>\n"
    "\t-r <random seed to generate the matrices>\n"
    "\t-s <input smart-kernel config file name>\n"
    "\t-t <number of tests for each kernel>\n"
    "\t-u <GPU ID in [0, #GPUs - 1]>\n"
    "\t-y <verbosity level in [0,2]>\n";
  float *A, *x, *y;  /* y = A * x */
  int M = 0, N = 0;  /* A is M x N, x is N x 1, y is M x 1 */  
  unsigned int randSeed; /* custom seed to random number generator */
  double ysumGpu; /* sum of all elements in y, the result vector */
  double ysumCpu; /* sum of all elements in y, the result vector */
  double ysumAbsDiff;
  double ysumRelDiff;
  double medianGFLOPS;
  bool runCpu = false;
  int  numGpus;
  cudaDeviceProp  prop;
  char *sgemvKernelConfigFileName = NULL;
  char *sgemvKernelConfigOutputFileName = NULL;
  size_t          freeBytesStart;
  size_t          totalBytesStart;
  size_t          freeBytesEnd;
  size_t          totalBytesEnd;
  int             option;
  int             gpuId = 0;
  int             trials = 5;
  int             verbosity = 1;
  
  /* get program options */
  while ((option = getopt(argc, argv, ":chM:N:o:r:s:t:u:y:")) != -1) {
    switch (option) {
    case 'c':
      runCpu = true;
      break;
      
    case 'h':
      fprintf(stderr, "Help:\n%s\n", helpMsg);
      exit(EXIT_FAILURE);
      break;
    
    case 'M':
      M = atoi(optarg);      
      break;

    case 'N':
      N = atoi(optarg);      
      break;
    
    case 'o':
      sgemvKernelConfigOutputFileName = optarg;
      break;

    case 'r':
      /* optional arg for setting the random seed */
      randSeed = atoi(optarg);      
      /* set random seed */
      srand(randSeed);
      if (verbosity >= 2)
        fprintf(stderr, "Info: random seed = %u\n", randSeed);
      break;
    
    case 's':
      sgemvKernelConfigFileName = optarg;
      break;

    case 't':
      trials = atoi(optarg);
      break;

    case 'u':
      gpuId = atoi(optarg);
      break;

    case 'y':
      verbosity = atoi(optarg);
      break;

    case ':':
      fprintf(stderr, "Option -%c requires an argument\n", optopt);
      exit(EXIT_FAILURE);
      break;

    case '?':
      fprintf(stderr, "Unknown option character '%c'.\n", optopt);
      exit(EXIT_FAILURE);
    }
  }
  if (M == 0 || N == 0) {
    fprintf(stderr, "M and N cannot be zero\n");
    exit(EXIT_FAILURE);
  }
  HANDLE_ERROR(cudaGetDeviceCount(&numGpus));
  for (int i = 0; i < numGpus; ++i) {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    if (verbosity >= 3) {
      fprintf(stderr, "Device %d:\n", i);
      hmlDevicePropertyPrint(&prop);
    }
    hmlDevicePropertyCheck(&prop);    
  }
  /* choose which device to run the kernel code */
  if (gpuId >= numGpus) {
    fprintf(stderr, "Warning: Invalid GPU card #%d, resetting to default (0)\n",
            gpuId);
    gpuId = 0;
  }  
  HANDLE_ERROR(cudaSetDevice(gpuId));
  if (verbosity >= 2)
    fprintf(stderr, "Set device to GPU #%d\n", gpuId);

  /* initialize SGEMV kernel */
  hmlSgemvKernelInit();  
  /* read sGEMV kernel config file, or create on-line config */  
  if (sgemvKernelConfigFileName)
    hmlSgemvKernelConfigReadFile(sgemvKernelConfigFileName);
  else
    hmlSgemvKernelConfigOnline(cHmlSgemvConfigTestMxNMax,
                            cHmlSgemvConfigTestsPerKernel,
                            1);  
  /* write the current kernel-selection config out to a file */
  if (sgemvKernelConfigOutputFileName) {
    if (sgemvKernelConfigFileName) {
      fprintf(stderr, 
        "Warning: Both input and output config files are specified\n");
    }
    FILE *file = openFile(sgemvKernelConfigOutputFileName, "wb");
    hmlSgemvKernelConfigPrint(file);
    fclose(file);
  }
  /* alloc memory for A, B, and C arrays */
  A = (float*)malloc(M * N * sizeof(float));  
  x = (float*)malloc(N * sizeof(float));  
  y = (float*)malloc(M * sizeof(float));
  /* everything is in row-major order */
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++)
      A[i*N + j] = hmlRandomFloat();
  for(int j = 0; j < N; j++)
      x[j] = hmlRandomFloat();
  for(int i = 0; i < M; i++)
      y[i] = hmlRandomFloat();

  /* get free and total memory statistics before */
  HANDLE_ERROR(cudaMemGetInfo(&freeBytesStart, &totalBytesStart));
  if (verbosity >= 2) {
    fprintf(stderr, "Free memory = %ld bytes, total memory = %ld bytes\n",
	    freeBytesStart, totalBytesStart);
  }
  /* run the basic sGEMV test */
  hmlSgemvBasicTest(y, A, x, M, N, 1.0, 0.0, trials, &medianGFLOPS);
  fprintf(stderr, "Basic kernel median GFLOPS = %.2lf\n", medianGFLOPS);
  /* run the smart sGEMV test */
  hmlSgemvTest(y, A, x, M, N, 1.0, 0.0, trials, &medianGFLOPS);
  fprintf(stderr, "Smart kernel median GFLOPS = %.2lf\n", medianGFLOPS);
  /* get free and total memory statistics after */
  HANDLE_ERROR(cudaMemGetInfo(&freeBytesEnd, &totalBytesEnd));
  if (verbosity >= 2) {
    fprintf(stderr, "Free memory = %ld bytes, total memory = %ld bytes\n",
	    freeBytesEnd, totalBytesEnd);
    /* report any potential memory leaks */
    if (freeBytesStart != freeBytesEnd || totalBytesStart != totalBytesEnd)
      fprintf(stderr, "Warning: Memory leak %ld bytes, %ld total bytes\n",
	      freeBytesStart - freeBytesEnd, totalBytesStart - totalBytesEnd);
  }
  ysumGpu = hmlComputeYsum(y, M);
  fprintf(stdout, "GPU ysum = %.20lf\n", ysumGpu);
  if (runCpu) {
    hmlSgemvCpu(y, A, x, M, N, 1.0, 0.0);
    ysumCpu = hmlComputeYsum(y, M);
    fprintf(stdout, "CPU ysum = %.20lf\n", ysumCpu);
    ysumAbsDiff = fabs(ysumGpu - ysumCpu);
    fprintf(stdout, "ysum absolute difference: %.20lf\n", ysumAbsDiff);
    ysumRelDiff = ysumAbsDiff / ysumCpu;
    fprintf(stdout, "ysum relative difference: %.20lf\n", ysumRelDiff);
    if (ysumRelDiff > cHmlSgemvRelativeErrorMax) {
      fprintf(stderr, "; Error: CPU and GPU results differ\n");
      return 1;
    }
  }

  return 0;
}
