/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_sgemm_kernel.h"
#include "hml_sgemm_kernel_template.h"

void
initSGemmKernel7(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1])
{
  /* set the variable-K kernels with 7 column stops */
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<7, 4, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<7, 5, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<7, 6, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 7, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 8, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 9, cHmlUseTextureMem>(varK);   
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 10, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 11, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 12, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 13, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 14, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 15, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<7, 16, cHmlUseTextureMem>(varK);
#endif

#ifdef HML_USE_CONST_K_KERNELS
  /* set the constant-K kernels with 7 column stops */
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 7, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 7, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 7, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 7, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 7, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 7, 16, cHmlUseTextureMem>(constK);
#endif
#endif /* HML_USE_CONST_K_KERNELS */
}
