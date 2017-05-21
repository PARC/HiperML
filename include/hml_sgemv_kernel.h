/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#ifndef HML_SGEMV_KERNELS_H_INCLUDED_
#define HML_SGEMV_KERNELS_H_INCLUDED_

#include "hml_common.h"

/* enable the use of texture memory, which runs faster on most GPUs,
 * but may fail to run on cards like K2000
 */
#define HML_USE_TEXTURE_MEM

#ifdef HML_USE_TEXTURE_MEM
#define cHmlUseTextureMem   1
#else
#define cHmlUseTextureMem   0
#endif /* HML_USE_TEXTURE_MEM */

/* enable the use of const-K kernels, which take longer to compile
 * but also runs faster than variable-K kernels that are always enabled
 */
#define HML_USE_CONST_N_KERNELS

/* SGEMV constants */
#define cHmlMaxSkinnyN      1024  /* max value of skinny N */
#define cHmlMaxStops        16    /* max # of row or column stops in stride 16 */
#define cHmlMaxBlockStops   16    /* max # of block stops in stride 16 */

#define cHmlMaxTestM     1000000  /* max value of N for speed test */
#define cHmlMaxTestN        8192  /* max value of N for speed test */

#define cHmlSgemvKernelBasicStops      1  /* default rowStops of base kernel */
#define cHmlSgemvKernelBasicBlockStops 8  /* default blockStops of base kernel */

typedef enum {
  eHmlSgemvKernelBasic = 0,
  eHmlSgemvKernelVarN,
  eHmlSgemvKernelConstN,
  cHmlSgemvKernelTypes
} HmlSgemvKernelType;

typedef struct {
  HmlSgemvKernelType type;
  int             blockStops;
  int             rowStops;
} HmlSgemvKernelConfig;

typedef void (*HmlSgemvKernelVarN)(float       *y,
                                   const float *A,
                                   const float *x,
                                   const int    M,
                                   const int    N,
                                   const float  alpha,
                                   const float  beta);

typedef void (*HmlSgemvKernelConstN)(float       *y,
                                     const float *A,
                                     const float *x,
                                     const int    M,
                                     const float  alpha,
                                     const float  beta);

typedef struct {
  HmlSgemvKernelVarN   basic[cHmlMaxBlockStops+1];
  HmlSgemvKernelVarN   varN[cHmlMaxBlockStops+1][cHmlMaxStops+1];
  HmlSgemvKernelConstN constN[cHmlMaxSkinnyN+1][cHmlMaxBlockStops+1][cHmlMaxStops+1];
} HmlSgemvKernelRepo;

void hmlSgemvKernelArgSet(HmlKernelArg *arg,
                          int        M,
                          int        N,
                          int        K,
                          int        colStops,
                          int        rowStops);

/* uses global array: hmlSgemvKernelConfig[] to pick the best
 * SGEMV kernel
 */
void
hmlSgemvKernelSelect(HmlSgemvKernelVarN   *varN,
                     HmlSgemvKernelConstN *constN,
                     HmlKernelArg         *karg,
                     const int          M,
                     const int          N);

void
hmlSgemvKernelSelectBasic(HmlSgemvKernelVarN   *varN,
                          HmlKernelArg         *karg,
                          const int          M,
                          const int          N,
                          const int          blockStops);

/* Single-precision general matrix multiplication (sGEMV) kernel.
 * matrix A is not transposed, and thus 'N' function name suffix
 * This kernel does not use shared memory.
 */
__global__ void
hmlSgemvKernel1T1RGmemN(float       *y,
                        const float *A,
                        const float *x,
                        const int    M,
                        const int    N,
                        const float  alpha,
                        const float  beta);

/* Single-precision general matrix multiplication (sGEMV) kernel.
 * matrix A is not transposed, and thus 'N' function name suffix
 * This kernel use shared memory.
 */
__global__ void
hmlSgemvKernel1T1RSmemN(float       *y,
                        const float *A,
                        const float *x,
                        const int    M,
                        const int    N,
                        const float  alpha,
                        const float  beta);

__global__ void
hmlSgemvKernel128T1RN(float       *y,
                      const float *A,
                      const float *x,
                      const int    M,
                      const int    N,
                      const float  alpha,
                      const float  beta);

void hmlSgemvKernelInit(void);

void hmlSgemvKernelConfigOnline(int maxMxN,
                                int testTrialsPerKernel,
                                int verbosity);

void hmlSgemvKernelConfigReadFile(const char *fileName);

void hmlSgemvKernelConfigPrint(FILE *file);

#endif /* HML_SGEMV_KERNELS_H_INCLUDED_ */
