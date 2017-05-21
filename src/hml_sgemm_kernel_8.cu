/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_sgemm_kernel.h"
#include "hml_sgemm_kernel_template.h"

void
initSGemmKernel8(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1])
{
  /* set the variable-K kernels with 8 column stops */
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<8, 4, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<8, 5, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<8, 6, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 7, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 8, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 9, cHmlUseTextureMem>(varK);   
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 10, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 11, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 12, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 13, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 14, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 15, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<8, 16, cHmlUseTextureMem>(varK);
#endif

#ifdef HML_USE_CONST_K_KERNELS
  /* set the constant-K kernels with 8 column stops */
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 8, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 8, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 8, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 8, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 8, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 8, 16, cHmlUseTextureMem>(constK);
#endif
#endif /* HML_USE_CONST_K_KERNELS */
}
