/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_sgemm_kernel.h"
#include "hml_sgemm_kernel_template.h"

void
initSGemmKernel6(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]) {
  /* set the variable-K kernels with 6 column stops */
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 4, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 5, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 6, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 7, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 8, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 9, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 10, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 11, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 12, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 13, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 14, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 15, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<6, 16, cHmlUseTextureMem>(varK);
#endif

#ifdef HML_USE_CONST_K_KERNELS
  /* set the constant-K kernels with 6 column stops */
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 6, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 6, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 6, 16, cHmlUseTextureMem>(constK);
#endif
#endif /* HML_USE_CONST_K_KERNELS */
}
