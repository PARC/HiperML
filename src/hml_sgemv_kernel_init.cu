/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include "hml_sgemv_kernel.h"
#include "hml_sgemv_kernel_template.h"

/* Some compute capabilities (e.g., 2.0) would terminate
 * kernels that use too many registers per thread, causing
 * the on-line config routine to believe the kernel runs
 * extremely "fast." Calling this function ensures the
 * integrity of the SGEMV kernel repository.
 * Note: the basic kernels are not affected by this function,
 * since its register usage doesn't grow with blockStops
 */
void
hmlSgemvKernelInitReset(HmlSgemvKernelVarN   varN[][cHmlMaxStops+1],
                        HmlSgemvKernelConstN constN[][cHmlMaxBlockStops+1][cHmlMaxStops+1])
{
  int blockStops;
  int rowStops;
  int N;
  cudaDeviceProp prop;

  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  for (blockStops = 1; blockStops <= cHmlMaxBlockStops; ++blockStops) {
    for (rowStops = 1; rowStops <= cHmlMaxStops; ++rowStops) {
      /* the following inequality test is only a heuristic!
       * the reason to use 'rowStop * 2' as a heuristic estimate of
       * how many registers will be used in the constN and varN kernels
       * is that both 'float yVal[rowStops];' and 'float AsCache[rowStops];'
       * need 'rowStops' 32-bit registers, although the actual number of
       * registers compiled by nvcc may be different
       */
      if (rowStops * 2 > hmlMaxNumRegistersPerThread(&prop)) {
        varN[blockStops][rowStops] = NULL;
        for (N = 1; N <= cHmlMaxSkinnyN; ++N)
          constN[N][blockStops][rowStops] = NULL;
      }
    }
  }
}

void
hmlSgemvKernelRepoInit(HmlSgemvKernelVarN  *basic,
                       HmlSgemvKernelVarN   varN[][cHmlMaxStops+1],
                       HmlSgemvKernelConstN constN[][cHmlMaxBlockStops+1][cHmlMaxStops+1])
{
  int numBasicKernels = cHmlMaxBlockStops + 1;
  int numVarNKernels = (cHmlMaxBlockStops + 1) * (cHmlMaxStops + 1);
  int numConstNKernels =
    (cHmlMaxSkinnyN + 1) * (cHmlMaxBlockStops + 1) * (cHmlMaxStops + 1);

  memset(basic, 0, sizeof(HmlSgemvKernelVarN) * numBasicKernels);
  memset(varN, 0, sizeof(HmlSgemvKernelVarN) * numVarNKernels);
  memset(constN, 0, sizeof(HmlSgemvKernelConstN) * numConstNKernels);

  /* using texture memory doesn't seem to help,
   * so we ignore cHmlUseTextureMem for basic kernels,
   * and pretend it is always false
   */
#if HML_SGEMV_BASIC_N_SMEM_BYTES(6) <= cHmlMaxSmemBytes
  hmlSgemvKernelBasicNSet<6, false>(basic);
#endif
#if HML_SGEMV_BASIC_N_SMEM_BYTES(7) <= cHmlMaxSmemBytes
  hmlSgemvKernelBasicNSet<7, false>(basic);
#endif
#if HML_SGEMV_BASIC_N_SMEM_BYTES(8) <= cHmlMaxSmemBytes
  hmlSgemvKernelBasicNSet<8, false>(basic);
#endif
#if HML_SGEMV_BASIC_N_SMEM_BYTES(9) <= cHmlMaxSmemBytes
  hmlSgemvKernelBasicNSet<9, false>(basic);
#endif
#if HML_SGEMV_BASIC_N_SMEM_BYTES(10) <= cHmlMaxSmemBytes
  hmlSgemvKernelBasicNSet<10, false>(basic);
#endif

#if HML_SGEMV_VAR_N_SMEM_BYTES(4, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<4, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(5, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<5, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(6, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<6, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(7, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<7, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<8, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(9, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<9, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<10, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(11, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<11, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<12, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(13, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<13, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<14, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(15, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<15, 1, cHmlUseTextureMem>(varN);
#endif
#if HML_SGEMV_VAR_N_SMEM_BYTES(16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetVarNN<16, 1, cHmlUseTextureMem>(varN);
#endif

#ifdef HML_USE_CONST_N_KERNELS

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(1, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<1, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(2, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<2, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(3, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<3, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(4, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<4, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(5, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<5, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(6, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<6, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(7, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<7, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(8, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<8, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(9, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<9, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(10, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<10, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(11, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<11, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(12, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<12, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(13, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<13, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(14, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<14, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(15, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<15, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(16, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<16, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(17, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<17, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(18, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<18, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(19, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<19, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(20, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<20, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(21, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<21, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(22, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<22, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(23, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<23, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(24, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<24, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(25, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<25, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(26, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<26, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(27, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<27, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(28, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<28, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(29, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<29, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(30, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<30, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(31, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<31, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(32, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<32, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(33, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<33, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(34, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<34, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(35, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<35, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(36, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<36, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(37, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<37, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(38, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<38, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(39, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<39, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(40, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<40, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(41, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<41, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(42, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<42, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(43, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<43, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(44, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<44, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(45, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<45, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(46, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<46, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(47, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<47, 16, 4, cHmlUseTextureMem>(constN);
#endif

#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 8, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 8, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 8, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 8, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 8, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 8, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 10, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 10, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 10, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 10, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 10, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 10, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 12, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 12, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 12, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 12, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 12, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 12, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 14, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 14, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 14, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 14, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 14, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 14, 4, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 16, 1) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 16, 1, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 16, 2) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 16, 2, cHmlUseTextureMem>(constN);
#endif
#if HML_SGEMV_CONST_N_N_SMEM_BYTES(48, 16, 4) <= cHmlMaxSmemBytes
  hmlSgemvKernelSetConstNN<48, 16, 4, cHmlUseTextureMem>(constN);
#endif

#endif /* HML_USE_CONST_N_KERNELS */

  hmlSgemvKernelInitReset(varN, constN);
}
