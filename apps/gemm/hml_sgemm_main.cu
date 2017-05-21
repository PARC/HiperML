/*
 * gemm.cu: GEneral Matrix-Matrix multiplication (GEMM) between two matrices
 * A and B, where A is M rows x K columns and B is K rows x N columns. 
 * Unlike CUBLAS and MAGMA, which use column-major store, this program assumes
 * both A and B are stored in row-major order.
 * This implementation is based on the paper:
 * "An Improved Magma Gemm for Fermi Graphics Processing Units"
 * by R. Nath, S. Tomov, and J. Dongarra published in 2011 at the International
 * Journal of High Performance Computing Applications 24(4) 511-515.
 *
 * The parameters used are: N_TBX = 96, N_TBY = 96, N_TX = 64, N_TY = 4, nb = 16,
 * where the definitions of these parameters can be found in the paper:
 * http://cseweb.ucsd.edu/~rknath/fermi_gemm.pdf
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
#include "hml_sgemm.h"
#include "hml_sgemm_kernel.h"

#define cHmlSgemmRelativeErrorMax         (1.0e-8)

#ifdef WIN32
#define cHmlSgemmConfigTestNumRows        100000
#define cHmlSgemmConfigTestsPerKernel 3
#else
#define cHmlSgemmConfigTestNumRows        1000000
#define cHmlSgemmConfigTestsPerKernel 5
#endif /* WIN32 */

#ifdef HML_USE_TEXTURE_MEM
/* external global variable declaration */
extern texture<float, 1> texFloatA;
extern texture<float, 1> texFloatB;
#endif /* HML_USE_TEXTURE_MEM */

/* Basic, non-smart API function for single precision GEMM (sGEMM)
 * for computing C = alpha * A * B + beta * C on the GPU
 * Each thread of the non-smart kernel computes 6 x 6 elements of
 * C with stride 16 (which is why it is called sgemm6x6NN).
 * Note: all matrices should not have any storage paddings. 
 */
void 
hmlSgemmTestBasic(float       *C,
	   const float *A,
	   const float *B,
	   int          M,
	   int          N,
	   int          K,
	   float        alpha,
	   float        beta,
	   int          trials,
	   double      *medianGFLOPS) 
{
  double cpuStart, wallStart;   /* time keeping variables */
  double cpuEnd, wallEnd;   /* time keeping variables */
  /* compute the GFLOPS based on sizes of the matrices */
  double gflop =      FLOPS_SGEMM(M, N, K) / 1e9;
  double              GFLOPS;
  std::vector<double> trialGFLOPS;
  float *devA, *devB, *devC; /* matrices on CUDA device */
  HmlSgemmKernelVarK     sgemmBasicKernel;
  HmlKernelArg           karg;
  cudaError_t         err;

#ifdef HML_USE_TEXTURE_MEM
  /* cHmlMaxCudaTexture1DLinear is in terms of elements instead of bytes */
  if (M * K >= cHmlMaxCudaTexture1DLinear || N * K >= cHmlMaxCudaTexture1DLinear) {
    fprintf(stderr, "CUDA maxTexture1DLinear exceeded\n");
    exit(1);
  }
#endif /* HML_USE_TEXTURE_MEM */

  /* alloc float array on device, load from host, and possibly
   * bind the device array to texture memory for A & B 
   */
#ifdef HML_USE_TEXTURE_MEM 
  devA = hmlDeviceFloatArrayAllocLoadBind(A, M * K, texFloatA);
  devB = hmlDeviceFloatArrayAllocLoadBind(B, N * K, texFloatB);
#else
  devA = hmlDeviceFloatArrayAllocLoad(A, M * K);
  devB = hmlDeviceFloatArrayAllocLoad(B, N * K);
#endif /* HML_USE_TEXTURE_MEM */
  
  /* alloc and load float array on device for C */
  devC = hmlDeviceFloatArrayAllocLoad(C, M * N);
  
  /* select basic kernel, and setup the kernel parameters */
  hmlSgemmKernelSelectBasic(&sgemmBasicKernel, &karg, M, N, K);
  for (int t = 0; t < trials; ++t) {
    hmlGetSecs(&cpuStart, &wallStart);  /* start the timer */
    /* invoke sGEMM kernal to compute C = alpha x A x B + beta x C */
    sgemmBasicKernel<<<karg.grid, karg.block>>>(devC,
						devA,
						devB,
						M,
						N,
						K,
						alpha,
						beta);
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
  /* copy C from device memory to host memory */
  err = cudaMemcpy(C, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Copy C off of device: %s\n",cudaGetErrorString(err));
    exit(1);
  }
  
#ifdef HML_USE_TEXTURE_MEM
  /* unbind texture memory */
  cudaUnbindTexture(texFloatA);
  cudaUnbindTexture(texFloatB);
#endif /* HML_USE_TEXTURE_MEM */
  
  /* free device memory */
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
}

/* The "smart" version of the API function for single precision GEMM
 * (sGEMM) for computing C = alpha * A * B + beta * C,
 * where it is the best if matrix A is tall and skinny.
 * Note: all matrices should not have any storage paddings. 
 */
void 
hmlSgemmTest(float       *C,
	  const float *A,
	  const float *B,
	  int          M,
	  int          N,
	  int          K,
	  float        alpha,
	  float        beta,
	  int          trials,
	  double      *medianGFLOPS) 
{
  double cpuStart, wallStart;   /* time keeping variables */
  double cpuEnd, wallEnd;   /* time keeping variables */
  /* compute the GFLOPS based on sizes of the matrices */
  double gflop =      FLOPS_SGEMM(M, N, K) / 1e9;
  double              GFLOPS;
  std::vector<double> trialGFLOPS;
  float *devA, *devB, *devC; /* matrices on CUDA device */    
  HmlSgemmKernelConstK   constK;
  HmlSgemmKernelVarK     varK;
  HmlKernelArg           karg;
  cudaError_t         err;

#ifdef HML_USE_TEXTURE_MEM
  /* cHmlMaxCudaTexture1DLinear is in terms of elements instead of bytes */
  if (M * K >= cHmlMaxCudaTexture1DLinear || N * K >= cHmlMaxCudaTexture1DLinear) {
    fprintf(stderr, "CUDA maxTexture1DLinear exceeded\n");
    exit(1);
  }
#endif /* HML_USE_TEXTURE_MEM */
  
  /* alloc float array on device, load from host, and possibly
   * bind the device array to texture memory for A & B 
   */
#ifdef HML_USE_TEXTURE_MEM 
  devA = hmlDeviceFloatArrayAllocLoadBind(A, M * K, texFloatA);
  devB = hmlDeviceFloatArrayAllocLoadBind(B, N * K, texFloatB);
#else
  devA = hmlDeviceFloatArrayAllocLoad(A, M * K);
  devB = hmlDeviceFloatArrayAllocLoad(B, N * K);
#endif /* HML_USE_TEXTURE_MEM */
  
  /* alloc and load float array on device for C */
  devC = hmlDeviceFloatArrayAllocLoad(C, M * N);

  /* pick the right kernel to run */
  hmlSgemmKernelSelect(&varK, &constK, &karg, M, N, K);
  for (int t = 0; t < trials; ++t) {
    hmlGetSecs(&cpuStart, &wallStart);   /* start the timer */
    /* invoke sGEMM kernal to compute C = alpha x A x B + beta x C */
    if (constK) {
      constK<<<karg.grid, karg.block>>>(devC,
					devA,
					devB,
					M,
					N,
					alpha,
					beta);
    }
    else {
      varK<<<karg.grid, karg.block>>>(devC,
				      devA,
				      devB,
				      M,
				      N,
				      K,
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
  /* copy C from device memory to host memory */
  err = cudaMemcpy(C, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Copy C off of device: %s\n",cudaGetErrorString(err));
    exit(1);
  }
  
#ifdef HML_USE_TEXTURE_MEM
  /* unbind texture memory */
  cudaUnbindTexture(texFloatA);
  cudaUnbindTexture(texFloatB);
#endif /* HML_USE_TEXTURE_MEM */

  /* free device memory */
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
}

/* test function for single precision GEMM (sGEMM) on CPU
 * NOTE: the dimensions of matrices do NOT have to multiples of 96
 * for best performance, it is desirable to have M, N, and K
 * as multiples of 32.
 * we assume row-major order.
 */
void
hmlSgemmCpu(float       *C,
	 const float *A,
	 const float *B,
         const int    M,
	 const int    N,
	 const int    K,
	 const float  alpha,
	 const float  beta)
{
  double cpuStart, wallStart;   /* time keeping variables */
  double cpuEnd, wallEnd;   /* time keeping variables */
  /* compute the GFLOPS based on sizes of the matrices */
  double gflop = FLOPS_SGEMM(M, N, K) / 1e9;    
  const float *colB;
  float  sum;
  
  hmlGetSecs(&cpuStart, &wallStart);   /* start the timer */
  for (int i = 0; i < M; ++i, A += K, C += N) {        
    for (int j = 0; j < N; ++j) {
      colB = B + j;
      sum = 0.0;
      for (int e = 0; e < K; ++e) {
        sum += A[e] * colB[e * N];
      }
      C[j] = alpha * sum + beta * C[j];
    }
  }
  hmlGetSecs(&cpuEnd, &wallEnd); /* stop the timer */
  fprintf(stderr, "CPU time = %lf, wall time = %lf\n",
          cpuEnd - cpuStart, wallEnd - wallStart);
  fprintf(stderr, "GFLOPS = %lf\n", gflop / (wallEnd - wallStart));  
}

double hmlComputeCsum(float *C, int M, int N)
{
  double Csum = 0.0;
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++) {
      double Cval = (double)(C[i*N + j]);
      Csum += Cval;
    }     
  }
  
  return Csum;
}

int main(int argc, char* argv[]) 
{
  char                       helpMsg[] =
    "This program does GPU-based Single-precision "
    "GEneral Matrix-Matrix multiplication (SGEMM):\n"
    "C = alpha * A * B + beta * C\n"
    "where A is M x K, B is K x N, and C is M x N\n"
    "Options:\n"
    "\t-c Cross check with CPU-based SGEMM results\n"
    "\t-h Print this help message\n"
    "\t-K <K>\n"
    "\t-M <M>\n"
    "\t-N <N>\n"
    "\t-o <output smart-kernel config file name>\n"
    "\t-r <random seed to generate the matrices>\n"
    "\t-s <input smart-kernel config file name>\n"
    "\t-t <number of tests for each kernel>\n"
    "\t-u <GPU ID in [0, #GPUs - 1]>\n"
    "\t-y <verbosity level in [0,2]>\n";
  float *A, *B, *C; /* C = A x B */
  int M = 0, N = 0, K = 0;     /* A is M x K, B is K x N, C is M x N */  
  unsigned int randSeed; /* custom seed to random number generator */
  double CsumGpu; /* sum of all elements in C, the result matrix */
  double CsumCpu; /* sum of all elements in C, the result matrix */
  double CsumAbsDiff;
  double CsumRelDiff;
  double medianGFLOPS;
  bool runCpu = false;
  int  numGpus;
  cudaDeviceProp  prop;
  char *sgemmKernelConfigFileName = NULL;
  char *sgemmKernelConfigOutputFileName = NULL;
  size_t          freeBytesStart;
  size_t          totalBytesStart;
  size_t          freeBytesEnd;
  size_t          totalBytesEnd;
  int             option;
  int             gpuId = 0;
  int             trials = 5;
  int             verbosity = 1;
  
  /* get program options */
  while ((option = getopt(argc, argv, ":chK:M:N:o:r:s:u:y:")) != -1) {
    switch (option) {
    case 'c':
      runCpu = true;
      break;
      
    case 'h':
      fprintf(stderr, "Help:\n%s\n", helpMsg);
      exit(EXIT_FAILURE);
      break;
    
    case 'K':
      K = atoi(optarg);      
      break;
    
    case 'M':
      M = atoi(optarg);      
      break;

    case 'N':
      N = atoi(optarg);      
      break;
    
    case 'o':
      sgemmKernelConfigOutputFileName = optarg;
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
      sgemmKernelConfigFileName = optarg;
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
  if (M == 0 || N == 0 || K == 0) {
    fprintf(stderr, "M, N, and K cannot be zero\n");
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

  /* initialize SGEMM kernel repository */
  hmlSgemmKernelRepoInit();
  
  /* read sGEMM kernel config file, or create on-line config */  
  if (sgemmKernelConfigFileName)
    hmlSgemmKernelConfigReadFile(sgemmKernelConfigFileName);
  else
    hmlSgemmKernelConfigOnline(cHmlSgemmConfigTestNumRows,
			    cHmlSgemmConfigTestsPerKernel,
			    1);
  

  /* write the current kernel-selection config out to a file */
  if (sgemmKernelConfigOutputFileName) {
    if (sgemmKernelConfigFileName) {
      fprintf(stderr, 
        "Warning: Both input and output config files are specified\n");
    }
    FILE *file = openFile(sgemmKernelConfigOutputFileName, "wb");
    hmlSgemmKernelConfigPrint(file);
    fclose(file);
  }
  
  /* alloc memory for A, B, and C arrays */
  A = (float*)malloc(M * K * sizeof(float));  
  B = (float*)malloc(N * K * sizeof(float));  
  C = (float*)malloc(M * N * sizeof(float));
  /* everything is in row-major order */
  for(int i = 0; i < M; i++)
    for(int j = 0; j < K; j++)
      A[i*K + j] = hmlRandomFloat();
  for(int i = 0; i < K; i++)
    for(int j = 0; j < N; j++)
      B[i*N + j] = hmlRandomFloat();
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++)
      C[i*N + j] = hmlRandomFloat();

 fprintf(stderr, "alloc done\n");
  /* get free and total memory statistics before */
  HANDLE_ERROR(cudaMemGetInfo(&freeBytesStart, &totalBytesStart));
  fprintf(stderr, "get info done\n");
  if (verbosity >= 2) {
    fprintf(stderr, "Free memory = %ld bytes, total memory = %ld bytes\n",
	    freeBytesStart, totalBytesStart);
  }
  /* run the basic sGEMM test */
  hmlSgemmTestBasic(C, A, B, M, N, K, 1.0, 0.0, trials, &medianGFLOPS);
  fprintf(stderr, "Basic kernel median GFLOPS = %.2lf\n", medianGFLOPS);
  /* run the smart sGEMM test */
  hmlSgemmTest(C, A, B, M, N, K, 1.0, 0.0, trials, &medianGFLOPS);
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
  /* print sum of all elements of C */
  CsumGpu = hmlComputeCsum(C, M, N);
  fprintf(stdout, "GPU Csum = %.20lf\n", CsumGpu);
  if (runCpu) {
    hmlSgemmCpu(C, A, B, M, N, K, 1.0, 0.0);
    CsumCpu = hmlComputeCsum(C, M, N);
    fprintf(stdout, "CPU Csum = %.20lf\n", CsumCpu);
    CsumAbsDiff = fabs(CsumGpu - CsumCpu);
    fprintf(stdout, "Csum absolute difference: %.20lf\n", CsumAbsDiff);
    CsumRelDiff = CsumAbsDiff / CsumCpu;
    fprintf(stdout, "Csum relative difference: %.20lf\n", CsumRelDiff);
    if (CsumRelDiff > cHmlSgemmRelativeErrorMax) {
      fprintf(stderr, "; Error: CPU and GPU results differ\n");
      return 1;
    }
  }

  return 0;

}
