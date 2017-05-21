/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_sgemm_kernel.h"
#include "hml_sgemm_kernel_template.h"

void
initSGemmKernel2(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]) {
  /* set the variable-K kernels with 2 column stops */
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 4, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 5, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 6, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 7, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 8, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 9, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 10, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 11, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 12, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 13, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 14, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 15, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<2, 16, cHmlUseTextureMem>(varK);
#endif

#ifdef HML_USE_CONST_K_KERNELS
  /* set the constant-K kernels with 2 column stops */
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 2, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 2, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 2, 16, cHmlUseTextureMem>(constK);
#endif
#endif /* HML_USE_CONST_K_KERNELS */
}
