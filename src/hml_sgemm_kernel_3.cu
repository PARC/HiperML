/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_sgemm_kernel.h"
#include "hml_sgemm_kernel_template.h"

void
initSGemmKernel3(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]) {
  /* set the variable-K kernels with 3 column stops */
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 4, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 5, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 6, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 7, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 8, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 9, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 10, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 11, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 12, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 13, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 14, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 15, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<3, 16, cHmlUseTextureMem>(varK);
#endif

#ifdef HML_USE_CONST_K_KERNELS
  /* set the constant-K kernels with 3 column stops */
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 3, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 4) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 4, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 5) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 5, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 6) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 6, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 7, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 8, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 9, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 10, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 11, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 3, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 3, 16, cHmlUseTextureMem>(constK);
#endif
#endif /* HML_USE_CONST_K_KERNELS */
}
