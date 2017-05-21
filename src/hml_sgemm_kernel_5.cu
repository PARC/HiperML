/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_sgemm_kernel.h"
#include "hml_sgemm_kernel_template.h"

void
initSGemmKernel5(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1])
{
  /* set the variable-K kernels with 5 column stops */
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<5, 4, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<5, 5, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<5, 6, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 7, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 8, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 9, cHmlUseTextureMem>(varK);   
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 10, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 11, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 12, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 13, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 14, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 15, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<5, 16, cHmlUseTextureMem>(varK);
#endif

#ifdef HML_USE_CONST_K_KERNELS
  /* set the constant-K kernels with 5 column stops */
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 5, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 5, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 5, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 5, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 5, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 5, 16, cHmlUseTextureMem>(constK);
#endif
#endif /* HML_USE_CONST_K_KERNELS */
}
