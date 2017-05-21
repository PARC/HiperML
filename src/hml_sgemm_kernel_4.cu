/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_sgemm_kernel.h"
#include "hml_sgemm_kernel_template.h"

void
initSGemmKernel4(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]) {
  /* set the variable-K kernels with 4 column stops */
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 4, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 5, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 6, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 7, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 8, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 9, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 10, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 11, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 12, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 13, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 14, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 15, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<4, 16, cHmlUseTextureMem>(varK);
#endif

#ifdef HML_USE_CONST_K_KERNELS
  /* set the constant-K kernels with 4 column stops */
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 4, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 4, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 4, 16, cHmlUseTextureMem>(constK);
#endif
#endif /* HML_USE_CONST_K_KERNELS */
}
