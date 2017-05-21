/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_sgemm_kernel.h"
#include "hml_sgemm_kernel_template.h"

void
initSGemmKernel1(
  HmlSgemmKernelVarK   varK[cHmlMaxStops+1][cHmlMaxStops+1],
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1])
{
  /* set the variable-K kernels with 1 column stop */
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<1, 4, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<1, 5, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelVarKNNSet<1, 6, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 7, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 8, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 9, cHmlUseTextureMem>(varK);   
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 10, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 11, cHmlUseTextureMem>(varK); 
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 12, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 13, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 14, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 15, cHmlUseTextureMem>(varK);
#endif
#if HML_SGEMM_VAR_K_NN_SMEM_BYTES(1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelVarKNNSet<1, 16, cHmlUseTextureMem>(varK);
#endif

#ifdef HML_USE_CONST_K_KERNELS
  /* set the constant-K kernels with 1 column stop */
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<2, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(2, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<2, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<3, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(3, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<3, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<4, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(4, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<4, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<5, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(5, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<5, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<6, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(6, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<6, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<7, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(7, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<7, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<8, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(8, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<8, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<9, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(9, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<9, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<10, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(10, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<10, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<11, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(11, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<11, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<12, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(12, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<12, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<13, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(13, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<13, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<14, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(14, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<14, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<15, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(15, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<15, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<16, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(16, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<16, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<17, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(17, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<17, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<18, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(18, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<18, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<19, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(19, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<19, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<20, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(20, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<20, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<21, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(21, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<21, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<22, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(22, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<22, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<23, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(23, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<23, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<24, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(24, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<24, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<25, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(25, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<25, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<26, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(26, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<26, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<27, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(27, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<27, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<28, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(28, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<28, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<29, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(29, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<29, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<30, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(30, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<30, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<31, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(31, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<31, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<32, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(32, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<32, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<33, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(33, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<33, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<34, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(34, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<34, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<35, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(35, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<35, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<36, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(36, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<36, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<37, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(37, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<37, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<38, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(38, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<38, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<39, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(39, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<39, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<40, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(40, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<40, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<41, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(41, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<41, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<42, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(42, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<42, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<43, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(43, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<43, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<44, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(44, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<44, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<45, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(45, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<45, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<46, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(46, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<46, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<47, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(47, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<47, 1, 16, cHmlUseTextureMem>(constK);
#endif

#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 4) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 1, 4, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 5) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 1, 5, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 6) <= cHmlMaxSmemBytes  
  hmlSgemmKernelConstKNNSet<48, 1, 6, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 7) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 7, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 8) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 8, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 9) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 9, cHmlUseTextureMem>(constK);   
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 10) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 10, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 11) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 11, cHmlUseTextureMem>(constK); 
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 12) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 12, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 13) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 13, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 14) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 14, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 15) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 15, cHmlUseTextureMem>(constK);
#endif
#if HML_SGEMM_CONST_K_NN_SMEM_BYTES(48, 1, 16) <= cHmlMaxSmemBytes
  hmlSgemmKernelConstKNNSet<48, 1, 16, cHmlUseTextureMem>(constK);
#endif
#endif /* HML_USE_CONST_K_KERNELS */
}
