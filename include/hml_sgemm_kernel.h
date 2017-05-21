/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_SGEMM_KERNEL_H_INCLUDED_
#define HML_SGEMM_KERNEL_H_INCLUDED_

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
#define HML_USE_CONST_K_KERNELS

/* SGEMM constants */
#define cHmlMaxSkinnyK                96  /* max value of skinny K */
#define cHmlMaxSkinnyN                96  /* max value of skinny N */
#define cHmlMaxStops                  16  /* max # of row or column stops */
#define cHmlSgemmKernelBasicStops      6  /* default value of row/column stops */

typedef enum {
  eHmlSgemmKernelBasic = 0,
  eHmlSgemmKernelVarK,
  eHmlSgemmKernelConstK,
  cHmlSgemmKernelTypes
} HmlSgemmKernelType;

typedef struct {
  HmlSgemmKernelType type;
  int             rowStops;
} HmlSgemmKernelConfig;

typedef void (*HmlSgemmKernelVarK)(float       *C,
                                   const float *A,
                                   const float *B,
                                   const int    M,
                                   const int    N,
                                   const int    K,
                                   const float  alpha,
                                   const float  beta);

typedef void (*HmlSgemmKernelConstK)(float       *C,
                                     const float *A,
                                     const float *B,
                                     const int    M,
                                     const int    N,
                                     const float  alpha,
                                     const float  beta);

typedef struct {
  HmlSgemmKernelVarK   basic;
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1];
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1];
} HmlSgemmKernelRepo;

void hmlSgemmKernelArgSet(HmlKernelArg *arg,
                          int           M,
                          int           N,
                          int           K,
                          int           colStops,
                          int           rowStops);

/* uses global array: hmlSgemmKernelConfig[][] to pick the best
 * SGEMM kernel
 */
void
hmlSgemmKernelSelect(HmlSgemmKernelVarK   *varK,
                     HmlSgemmKernelConstK *constK,
                     HmlKernelArg         *karg,
                     const int             M,
                     const int             N,
                     const int             K);

void
hmlSgemmKernelSelectBasic(HmlSgemmKernelVarK   *basic,
                          HmlKernelArg         *karg,
                          const int             M,
                          const int             N,
                          const int             K);

void hmlSgemmKernelRepoInit(void);

void hmlSgemmKernelConfigOnline(int testRows,
                                int testTrialsPerKernel,
                                int verbosity);

void hmlSgemmKernelConfigReadFile(const char *fileName);

void hmlSgemmKernelConfigPrint(FILE *file);

#endif /* HML_SGEMM_KERNEL_H_INCLUDED_ */
