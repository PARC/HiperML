/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

/*
 * hml_sgemm.cu: Single-precision
 * GEneral Matrix-Matrix multiplication (GEMM) between two matrices
 * A and B, where A is M rows x K columns and B is K rows x N columns.
 * Unlike CUBLAS and MAGMA, which use column-major store, this program assumes
 * both A and B are stored in row-major order.
 */
#include "hml_sgemm.h"
#include "hml_sgemm_kernel.h"

#ifdef WIN32
#define cHmlSgemmConfigTestRows       100000
#define cHmlSgemmConfigTestsPerKernel 3
#else
#define cHmlSgemmConfigTestRows       1000000
#define cHmlSgemmConfigTestsPerKernel 5
#endif /* WIN32 */

extern bool sgemmKernelRepoInitialized;
extern bool sgemmKernelConfigDone;

#ifdef HML_USE_TEXTURE_MEM
/* global texture variables */
texture<float, 1> texFloatA;
texture<float, 1> texFloatB;
#endif /* HML_USE_TEXTURE_MEM */

/* Basic, non-smart API function for single precision GEMM (sGEMM)
 * for computing C = alpha * A * B + beta * C on the GPU
 * Each thread of the non-smart kernel computes 6 x 6 elements of
 * C with stride 16 (which is why it is called sGemm6x6NN).
 * Note: all matrices should not have any storage paddings.
 */
void
hmlSgemmBasic(float       *C,
       const float *A,
       const float *B,
       const int    M,
       const int    N,
       const int    K,
       const float  alpha,
       const float  beta)
{
  float *devA, *devB, *devC; /* matrices on CUDA device */
  HmlSgemmKernelVarK sGemmBasicKernel;
  HmlKernelArg karg;

  /* initialize SGEMM kernel repository if not yet done */
  if (!sgemmKernelRepoInitialized)
    hmlSgemmKernelRepoInit();

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
  hmlSgemmKernelSelectBasic(&sGemmBasicKernel, &karg, M, N, K);
  /* invoke sGEMM kernal to compute C = alpha x A x B + beta x C */
  sGemmBasicKernel<<<karg.grid, karg.block>>>(devC,
                                              devA,
                                              devB,
                                              M,
                                              N,
                                              K,
                                              alpha,
                                              beta);
  /* block until computation is completed */
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Run kernel: %s\n", cudaGetErrorString(err));
    exit(1);
  }
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
hmlSgemm(float       *C,
      const float *A,
      const float *B,
      const int    M,
      const int    N,
      const int    K,
      const float  alpha,
      const float  beta)
{
  float *devA, *devB, *devC; /* matrices on CUDA device */
  HmlSgemmKernelConstK constK;
  HmlSgemmKernelVarK   varK;
  HmlKernelArg         karg;

  /* initialize SGEMM kernel repository if not yet done */
  if (!sgemmKernelRepoInitialized)
    hmlSgemmKernelRepoInit();

  /* perform on-line kernel config if no config has been done */
  if (!sgemmKernelConfigDone)
    hmlSgemmKernelConfigOnline(cHmlSgemmConfigTestRows,
                            cHmlSgemmConfigTestsPerKernel, 0);

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
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Run kernel: %s\n", cudaGetErrorString(err));
    exit(1);
  }
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

void
hmlSgemmKernelConfig(char *sgemmKernelConfigFileName)
{
  /* initialize SGEMM kernel repository if not yet done */
  if (!sgemmKernelRepoInitialized)
    hmlSgemmKernelRepoInit();

  if (sgemmKernelConfigFileName)
    hmlSgemmKernelConfigReadFile(sgemmKernelConfigFileName);
  else
    hmlSgemmKernelConfigOnline(cHmlSgemmConfigTestRows,
                            cHmlSgemmConfigTestsPerKernel, 0);
}
