/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_SGEMM_H_INCLUDED_
#define HML_SGEMM_H_INCLUDED_

#include "hml_common.h"

/* API function for configuring single precision GEMM (sGEMM)
 * kernel functions on the GPU
 */
void
hmlSgemmKernelConfig(char *sgemmKernelConfigFileName);

/* Basic, non-smart API function for single precision GEMM (sGEMM)
 * for computing C = alpha * A * B + beta * C on the GPU
 * Note: all matrices should not have any storage paddings.
 */
void hmlSgemmBasic(float       *C,
                   const float *A,
                   const float *B,
                   const int    M,
                   const int    N,
                   const int    K,
                   const float  alpha,
                   const float  beta);

/* The "smart" version of the API function for single precision GEMM
 * (sGEMM) for computing C = alpha * A * B + beta * C,
 * where it is the best if matrix A is tall and skinny.
 * Note: all matrices should not have any storage paddings.
 */
void hmlSgemm(float       *C,
              const float *A,
              const float *B,
              const int    M,
              const int    N,
              const int    K,
              const float  alpha,
              const float  beta);

#endif /* HML_SGEMM_H_INCLUDED_ */
