/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

/*
 * hml_sgemv.cu: Single-precision
 * GEneral Matrix-Matrix multiplication (GEMV) between two matrices
 * A and B, where A is M rows x K columns and B is K rows x N columns.
 * Unlike CUBLAS and MAGMA, which use column-major store, this program assumes
 * both A and B are stored in row-major order.
 */
#include "hml_sgemv.h"
#include "hml_sgemv_kernel.h"
#include "hml_flops.h"

#ifdef WIN32
#define cHmlSgemvConfigTestMxNMax       20000000
#define cHmlSgemvConfigTestsPerKernel   3
#else
#define cHmlSgemvConfigTestMxNMax       100000000
#define cHmlSgemvConfigTestsPerKernel   5
#endif /* WIN32 */

extern bool hmlSgemvKernelInitialized;
extern bool hmlSgemvKernelConfigDone;

#ifdef HML_USE_TEXTURE_MEM
/* global texture variables */
texture<float, 1> texFloatA;
texture<float, 1> texFloatx;
#endif /* HML_USE_TEXTURE_MEM */

/* Basic, non-smart API function for single precision GEMV (sGEMV)
 * for computing C = alpha * A * B + beta * C on the GPU
 * Note: all matrices should not have any storage paddings.
 */
void
hmlSgemvBasic(float       *y,
              const float *A,
              const float *x,
              const int    M,
              const int    N,
              const float  alpha,
              const float  beta) {
  float *devA, *devx, *devy; /* matrices on CUDA device */
  HmlSgemvKernelVarN sGemvBasicKernel;
  HmlKernelArg karg;

  /* initialize SGEMV kernel if not yet done */
  if(!hmlSgemvKernelInitialized) {
    hmlSgemvKernelInit();
  }

#ifdef HML_USE_TEXTURE_MEM
  /* cHmlMaxCudaTexture1DLinear is in terms of elements instead of bytes */
  if(M * N >= cHmlMaxCudaTexture1DLinear) {
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
  hmlSgemvKernelSelectBasic(&sGemvBasicKernel, &karg, M, N,
                            8/*cHmlSgemvKernelBasicBlockStops*/);
  /* invoke sGEMV basic kernal to compute y = alpha * A * x + beta * y */
  sGemvBasicKernel<<<karg.grid, karg.block>>>(devy,
      devA,
      devx,
      M,
      N,
      alpha,
      beta);
  /*
  if (M <= 8500)
    sGemv0GmemN<<<(M+127)/128, 128>>>(devy, devA, devx, M, N, alpha, beta);
  else
    sGemv0SmemN<<<(M+127)/128, 128>>>(devy, devA, devx, M, N, alpha, beta);
  */

  /* block until computation is completed */
  cudaError_t err = cudaDeviceSynchronize();
  if(err != cudaSuccess) {
    fprintf(stderr, "Run kernel: %s\n", cudaGetErrorString(err));
    exit(1);
  }
  /* copy y from device memory to host memory */
  err = cudaMemcpy(y, devy, M * sizeof(float), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
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
hmlSgemv(float       *y,
         const float *A,
         const float *x,
         const int    M,
         const int    N,
         const float  alpha,
         const float  beta) {
  float *devA, *devx, *devy; /* matrices on CUDA device */
  HmlSgemvKernelConstN constN;
  HmlSgemvKernelVarN   varN;
  HmlKernelArg         karg;

  /* initialize SGEMV kernel if not yet done */
  if(!hmlSgemvKernelInitialized) {
    hmlSgemvKernelInit();
  }

  /* perform on-line kernel config if no config has been done */
  if(!hmlSgemvKernelConfigDone)
    hmlSgemvKernelConfigOnline(cHmlSgemvConfigTestMxNMax,
                               cHmlSgemvConfigTestsPerKernel, 0);

#ifdef HML_USE_TEXTURE_MEM
  /* cHmlMaxCudaTexture1DLinear is in terms of elements instead of bytes */
  if(M * N >= cHmlMaxCudaTexture1DLinear) {
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
  /* invoke sGEMV kernal to compute y = alpha * A * x + beta * y */
  if(constN) {
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
  cudaError_t err = cudaDeviceSynchronize();
  if(err != cudaSuccess) {
    fprintf(stderr, "Run kernel: %s\n", cudaGetErrorString(err));
    exit(1);
  }
  /* copy vector y from device memory to host memory */
  err = cudaMemcpy(y, devy, M * sizeof(float), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
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

void
hmlSgemvKernelConfig(char *sgemvKernelConfigFileName) {
  /* initialize SGEMV kernel if not yet done */
  if(!hmlSgemvKernelInitialized) {
    hmlSgemvKernelInit();
  }

  if(sgemvKernelConfigFileName) {
    hmlSgemvKernelConfigReadFile(sgemvKernelConfigFileName);
  }
  else
    hmlSgemvKernelConfigOnline(cHmlSgemvConfigTestMxNMax,
                               cHmlSgemvConfigTestsPerKernel, 0);
}
