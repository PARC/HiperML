/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#ifndef HML_SGEMV_H_INCLUDED_
#define HML_SGEMV_H_INCLUDED_

/* API function for configuring single precision GEMV (sGEMV)
 * kernel functions on the GPU
 */
void
hmlSgemvKernelConfig(char *sgemvKernelConfigFileName);

/* Basic, non-smart API function for single precision GEMV (sGEMV)
 * for computing y = alpha * A * x + beta * y on the GPU
 * Note: matrix A and vectors x, y should not have any storage paddings.
 */
void hmlSgemvBasic(float       *y,
                   const float *A,
                   const float *x,
                   const int    M,
                   const int    N,
                   const float  alpha,
                   const float  beta);

/* The "smart" version of the API function for single precision GEMV
 * (sGEMV) for computing y = alpha * A * x + beta * y on the GPU,
 * where it is the best if matrix A is tall and skinny.
 * Note: matrix A and vectors x, y should not have any storage paddings.
 */
void hmlSgemv(float       *y,
              const float *A,
              const float *x,
              const int    M,
              const int    N,
              const float  alpha,
              const float  beta);

#endif /* HML_SGEMV_H_INCLUDED_ */
