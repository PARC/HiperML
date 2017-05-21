/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include <string.h>

#include "hml_kmeans_kernel.h"
#include "hml_kmeans_kernel_template.h"

/* init kernels that do not use texture memory */
void
hmlKmeansInitKernelAssignGmem(HmlKmeansAssignKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansAssignKernel) * (cHmlKmeansMaxDims + 1));
  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansAssignCtrdGmem<1, false>; */
  kernel[2] = hmlKmeansAssignCtrdGmem<2, false>;
  kernel[3] = hmlKmeansAssignCtrdGmem<3, false>;
  kernel[4] = hmlKmeansAssignCtrdGmem<4, false>;
  kernel[5] = hmlKmeansAssignCtrdGmem<5, false>;
  kernel[6] = hmlKmeansAssignCtrdGmem<6, false>;
  kernel[7] = hmlKmeansAssignCtrdGmem<7, false>;
  kernel[8] = hmlKmeansAssignCtrdGmem<8, false>;
  kernel[9] = hmlKmeansAssignCtrdGmem<9, false>;
  kernel[10] = hmlKmeansAssignCtrdGmem<10, false>;
  kernel[11] = hmlKmeansAssignCtrdGmem<11, false>;
  kernel[12] = hmlKmeansAssignCtrdGmem<12, false>;
  kernel[13] = hmlKmeansAssignCtrdGmem<13, false>;
  kernel[14] = hmlKmeansAssignCtrdGmem<14, false>;
  kernel[15] = hmlKmeansAssignCtrdGmem<15, false>;
  kernel[16] = hmlKmeansAssignCtrdGmem<16, false>;
  kernel[17] = hmlKmeansAssignCtrdGmem<17, false>;
  kernel[18] = hmlKmeansAssignCtrdGmem<18, false>;
  kernel[19] = hmlKmeansAssignCtrdGmem<19, false>;
  kernel[20] = hmlKmeansAssignCtrdGmem<20, false>;
  kernel[21] = hmlKmeansAssignCtrdGmem<21, false>;
  kernel[22] = hmlKmeansAssignCtrdGmem<22, false>;
  kernel[23] = hmlKmeansAssignCtrdGmem<23, false>;
  kernel[24] = hmlKmeansAssignCtrdGmem<24, false>;
  kernel[25] = hmlKmeansAssignCtrdGmem<25, false>;
  kernel[26] = hmlKmeansAssignCtrdGmem<26, false>;
  kernel[27] = hmlKmeansAssignCtrdGmem<27, false>;
  kernel[28] = hmlKmeansAssignCtrdGmem<28, false>;
  kernel[29] = hmlKmeansAssignCtrdGmem<29, false>;
  kernel[30] = hmlKmeansAssignCtrdGmem<30, false>;
  kernel[31] = hmlKmeansAssignCtrdGmem<31, false>;
  kernel[32] = hmlKmeansAssignCtrdGmem<32, false>;
  kernel[33] = hmlKmeansAssignCtrdGmem<33, false>;
  kernel[34] = hmlKmeansAssignCtrdGmem<34, false>;
  kernel[35] = hmlKmeansAssignCtrdGmem<35, false>;
  kernel[36] = hmlKmeansAssignCtrdGmem<36, false>;
  kernel[37] = hmlKmeansAssignCtrdGmem<37, false>;
  kernel[38] = hmlKmeansAssignCtrdGmem<38, false>;
  kernel[39] = hmlKmeansAssignCtrdGmem<39, false>;
  kernel[40] = hmlKmeansAssignCtrdGmem<40, false>;
  /*
  kernel[41] = hmlKmeansAssignCtrdGmem<41, false>;
  kernel[42] = hmlKmeansAssignCtrdGmem<42, false>;
  kernel[43] = hmlKmeansAssignCtrdGmem<43, false>;
  kernel[44] = hmlKmeansAssignCtrdGmem<44, false>;
  kernel[45] = hmlKmeansAssignCtrdGmem<45, false>;
  kernel[46] = hmlKmeansAssignCtrdGmem<46, false>;
  kernel[47] = hmlKmeansAssignCtrdGmem<47, false>;
  kernel[48] = hmlKmeansAssignCtrdGmem<48, false>;
  kernel[49] = hmlKmeansAssignCtrdGmem<49, false>;
  kernel[50] = hmlKmeansAssignCtrdGmem<50, false>;
  kernel[51] = hmlKmeansAssignCtrdGmem<51, false>;
  kernel[52] = hmlKmeansAssignCtrdGmem<52, false>;
  kernel[53] = hmlKmeansAssignCtrdGmem<53, false>;
  kernel[54] = hmlKmeansAssignCtrdGmem<54, false>;
  kernel[55] = hmlKmeansAssignCtrdGmem<55, false>;
  kernel[56] = hmlKmeansAssignCtrdGmem<56, false>;
  kernel[57] = hmlKmeansAssignCtrdGmem<57, false>;
  kernel[58] = hmlKmeansAssignCtrdGmem<58, false>;
  kernel[59] = hmlKmeansAssignCtrdGmem<59, false>;
  kernel[60] = hmlKmeansAssignCtrdGmem<60, false>;
  kernel[61] = hmlKmeansAssignCtrdGmem<61, false>;
  kernel[62] = hmlKmeansAssignCtrdGmem<62, false>;
  kernel[63] = hmlKmeansAssignCtrdGmem<63, false>;
  kernel[64] = hmlKmeansAssignCtrdGmem<64, false>;
  */
}

void
hmlKmeansInitKernelAssignSmem(HmlKmeansAssignKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansAssignKernel) * (cHmlKmeansMaxDims + 1));

  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansAssignCtrdSmem<1, false>; */
  kernel[2] = hmlKmeansAssignCtrdSmem<2, false>;
  kernel[3] = hmlKmeansAssignCtrdSmem<3, false>;
  kernel[4] = hmlKmeansAssignCtrdSmem<4, false>;
  kernel[5] = hmlKmeansAssignCtrdSmem<5, false>;
  kernel[6] = hmlKmeansAssignCtrdSmem<6, false>;
  kernel[7] = hmlKmeansAssignCtrdSmem<7, false>;
  kernel[8] = hmlKmeansAssignCtrdSmem<8, false>;
  kernel[9] = hmlKmeansAssignCtrdSmem<9, false>;
  kernel[10] = hmlKmeansAssignCtrdSmem<10, false>;
  kernel[11] = hmlKmeansAssignCtrdSmem<11, false>;
  kernel[12] = hmlKmeansAssignCtrdSmem<12, false>;
  kernel[13] = hmlKmeansAssignCtrdSmem<13, false>;
  kernel[14] = hmlKmeansAssignCtrdSmem<14, false>;
  kernel[15] = hmlKmeansAssignCtrdSmem<15, false>;
  kernel[16] = hmlKmeansAssignCtrdSmem<16, false>;
  kernel[17] = hmlKmeansAssignCtrdSmem<17, false>;
  kernel[18] = hmlKmeansAssignCtrdSmem<18, false>;
  kernel[19] = hmlKmeansAssignCtrdSmem<19, false>;
  kernel[20] = hmlKmeansAssignCtrdSmem<20, false>;
  kernel[21] = hmlKmeansAssignCtrdSmem<21, false>;
  kernel[22] = hmlKmeansAssignCtrdSmem<22, false>;
  kernel[23] = hmlKmeansAssignCtrdSmem<23, false>;
  kernel[24] = hmlKmeansAssignCtrdSmem<24, false>;
  kernel[25] = hmlKmeansAssignCtrdSmem<25, false>;
  kernel[26] = hmlKmeansAssignCtrdSmem<26, false>;
  kernel[27] = hmlKmeansAssignCtrdSmem<27, false>;
  kernel[28] = hmlKmeansAssignCtrdSmem<28, false>;
  kernel[29] = hmlKmeansAssignCtrdSmem<29, false>;
  kernel[30] = hmlKmeansAssignCtrdSmem<30, false>;
  kernel[31] = hmlKmeansAssignCtrdSmem<31, false>;
  kernel[32] = hmlKmeansAssignCtrdSmem<32, false>;
  kernel[33] = hmlKmeansAssignCtrdSmem<33, false>;
  kernel[34] = hmlKmeansAssignCtrdSmem<34, false>;
  kernel[35] = hmlKmeansAssignCtrdSmem<35, false>;
  kernel[36] = hmlKmeansAssignCtrdSmem<36, false>;
  kernel[37] = hmlKmeansAssignCtrdSmem<37, false>;
  kernel[38] = hmlKmeansAssignCtrdSmem<38, false>;
  kernel[39] = hmlKmeansAssignCtrdSmem<39, false>;
  kernel[40] = hmlKmeansAssignCtrdSmem<40, false>;
  /*
  kernel[41] = hmlKmeansAssignCtrdSmem<41, false>;
  kernel[42] = hmlKmeansAssignCtrdSmem<42, false>;
  kernel[43] = hmlKmeansAssignCtrdSmem<43, false>;
  kernel[44] = hmlKmeansAssignCtrdSmem<44, false>;
  kernel[45] = hmlKmeansAssignCtrdSmem<45, false>;
  kernel[46] = hmlKmeansAssignCtrdSmem<46, false>;
  kernel[47] = hmlKmeansAssignCtrdSmem<47, false>;
  kernel[48] = hmlKmeansAssignCtrdSmem<48, false>;
  kernel[49] = hmlKmeansAssignCtrdSmem<49, false>;
  kernel[50] = hmlKmeansAssignCtrdSmem<50, false>;
  kernel[51] = hmlKmeansAssignCtrdSmem<51, false>;
  kernel[52] = hmlKmeansAssignCtrdSmem<52, false>;
  kernel[53] = hmlKmeansAssignCtrdSmem<53, false>;
  kernel[54] = hmlKmeansAssignCtrdSmem<54, false>;
  kernel[55] = hmlKmeansAssignCtrdSmem<55, false>;
  kernel[56] = hmlKmeansAssignCtrdSmem<56, false>;
  kernel[57] = hmlKmeansAssignCtrdSmem<57, false>;
  kernel[58] = hmlKmeansAssignCtrdSmem<58, false>;
  kernel[59] = hmlKmeansAssignCtrdSmem<59, false>;
  kernel[60] = hmlKmeansAssignCtrdSmem<60, false>;
  kernel[61] = hmlKmeansAssignCtrdSmem<61, false>;
  kernel[62] = hmlKmeansAssignCtrdSmem<62, false>;
  kernel[63] = hmlKmeansAssignCtrdSmem<63, false>;
  kernel[64] = hmlKmeansAssignCtrdSmem<64, false>;
  */
}

void
hmlKmeansInitKernelAssignSmemUnroll(HmlKmeansAssignKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansAssignKernel) * (cHmlKmeansMaxDims + 1));

  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansAssignCtrdSmemUnroll<1, false>; */
  kernel[2] = hmlKmeansAssignCtrdSmemUnroll<2, false>;
  kernel[3] = hmlKmeansAssignCtrdSmemUnroll<3, false>;
  kernel[4] = hmlKmeansAssignCtrdSmemUnroll<4, false>;
  kernel[5] = hmlKmeansAssignCtrdSmemUnroll<5, false>;
  kernel[6] = hmlKmeansAssignCtrdSmemUnroll<6, false>;
  kernel[7] = hmlKmeansAssignCtrdSmemUnroll<7, false>;
  kernel[8] = hmlKmeansAssignCtrdSmemUnroll<8, false>;
  kernel[9] = hmlKmeansAssignCtrdSmemUnroll<9, false>;
  kernel[10] = hmlKmeansAssignCtrdSmemUnroll<10, false>;
  kernel[11] = hmlKmeansAssignCtrdSmemUnroll<11, false>;
  kernel[12] = hmlKmeansAssignCtrdSmemUnroll<12, false>;
  kernel[13] = hmlKmeansAssignCtrdSmemUnroll<13, false>;
  kernel[14] = hmlKmeansAssignCtrdSmemUnroll<14, false>;
  kernel[15] = hmlKmeansAssignCtrdSmemUnroll<15, false>;
  kernel[16] = hmlKmeansAssignCtrdSmemUnroll<16, false>;
  kernel[17] = hmlKmeansAssignCtrdSmemUnroll<17, false>;
  kernel[18] = hmlKmeansAssignCtrdSmemUnroll<18, false>;
  kernel[19] = hmlKmeansAssignCtrdSmemUnroll<19, false>;
  kernel[20] = hmlKmeansAssignCtrdSmemUnroll<20, false>;
  kernel[21] = hmlKmeansAssignCtrdSmemUnroll<21, false>;
  kernel[22] = hmlKmeansAssignCtrdSmemUnroll<22, false>;
  kernel[23] = hmlKmeansAssignCtrdSmemUnroll<23, false>;
  kernel[24] = hmlKmeansAssignCtrdSmemUnroll<24, false>;
  kernel[25] = hmlKmeansAssignCtrdSmemUnroll<25, false>;
  kernel[26] = hmlKmeansAssignCtrdSmemUnroll<26, false>;
  kernel[27] = hmlKmeansAssignCtrdSmemUnroll<27, false>;
  kernel[28] = hmlKmeansAssignCtrdSmemUnroll<28, false>;
  kernel[29] = hmlKmeansAssignCtrdSmemUnroll<29, false>;
  kernel[30] = hmlKmeansAssignCtrdSmemUnroll<30, false>;
  kernel[31] = hmlKmeansAssignCtrdSmemUnroll<31, false>;
  kernel[32] = hmlKmeansAssignCtrdSmemUnroll<32, false>;
  kernel[33] = hmlKmeansAssignCtrdSmemUnroll<33, false>;
  kernel[34] = hmlKmeansAssignCtrdSmemUnroll<34, false>;
  kernel[35] = hmlKmeansAssignCtrdSmemUnroll<35, false>;
  kernel[36] = hmlKmeansAssignCtrdSmemUnroll<36, false>;
  kernel[37] = hmlKmeansAssignCtrdSmemUnroll<37, false>;
  kernel[38] = hmlKmeansAssignCtrdSmemUnroll<38, false>;
  kernel[39] = hmlKmeansAssignCtrdSmemUnroll<39, false>;
  kernel[40] = hmlKmeansAssignCtrdSmemUnroll<40, false>;
  /*
  kernel[41] = hmlKmeansAssignCtrdSmemUnroll<41, false>;
  kernel[42] = hmlKmeansAssignCtrdSmemUnroll<42, false>;
  kernel[43] = hmlKmeansAssignCtrdSmemUnroll<43, false>;
  kernel[44] = hmlKmeansAssignCtrdSmemUnroll<44, false>;
  kernel[45] = hmlKmeansAssignCtrdSmemUnroll<45, false>;
  kernel[46] = hmlKmeansAssignCtrdSmemUnroll<46, false>;
  kernel[47] = hmlKmeansAssignCtrdSmemUnroll<47, false>;
  kernel[48] = hmlKmeansAssignCtrdSmemUnroll<48, false>;
  kernel[49] = hmlKmeansAssignCtrdSmemUnroll<49, false>;
  kernel[50] = hmlKmeansAssignCtrdSmemUnroll<50, false>;
  kernel[51] = hmlKmeansAssignCtrdSmemUnroll<51, false>;
  kernel[52] = hmlKmeansAssignCtrdSmemUnroll<52, false>;
  kernel[53] = hmlKmeansAssignCtrdSmemUnroll<53, false>;
  kernel[54] = hmlKmeansAssignCtrdSmemUnroll<54, false>;
  kernel[55] = hmlKmeansAssignCtrdSmemUnroll<55, false>;
  kernel[56] = hmlKmeansAssignCtrdSmemUnroll<56, false>;
  kernel[57] = hmlKmeansAssignCtrdSmemUnroll<57, false>;
  kernel[58] = hmlKmeansAssignCtrdSmemUnroll<58, false>;
  kernel[59] = hmlKmeansAssignCtrdSmemUnroll<59, false>;
  kernel[60] = hmlKmeansAssignCtrdSmemUnroll<60, false>;
  kernel[61] = hmlKmeansAssignCtrdSmemUnroll<61, false>;
  kernel[62] = hmlKmeansAssignCtrdSmemUnroll<62, false>;
  kernel[63] = hmlKmeansAssignCtrdSmemUnroll<63, false>;
  kernel[64] = hmlKmeansAssignCtrdSmemUnroll<64, false>;
  */
}

void
hmlKmeansInitKernelUpdateGmem(HmlKmeansUpdateKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansUpdateKernel) * (cHmlKmeansMaxDims + 1));

  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansUpdateCtrdGmem<1, 1, 0, false>; */
  kernel[2] = hmlKmeansUpdateCtrdGmem<2, 2, 0, false>;
  kernel[3] = hmlKmeansUpdateCtrdGmem<3, 3, 0, false>;
  kernel[4] = hmlKmeansUpdateCtrdGmem<4, 4, 0, false>;
  kernel[5] = hmlKmeansUpdateCtrdGmem<5, 5, 0, false>;
  kernel[6] = hmlKmeansUpdateCtrdGmem<6, 6, 0, false>;
  kernel[7] = hmlKmeansUpdateCtrdGmem<7, 7, 0, false>;
  kernel[8] = hmlKmeansUpdateCtrdGmem<8, 8, 0, false>;
  kernel[9] = hmlKmeansUpdateCtrdGmem<9, 9, 0, false>;
  kernel[10] = hmlKmeansUpdateCtrdGmem<10, 10, 0, false>;
  kernel[11] = hmlKmeansUpdateCtrdGmem<11, 11, 0, false>;
  kernel[12] = hmlKmeansUpdateCtrdGmem<12, 12, 0, false>;
  kernel[13] = hmlKmeansUpdateCtrdGmem<13, 13, 0, false>;
  kernel[14] = hmlKmeansUpdateCtrdGmem<14, 14, 0, false>;
  kernel[15] = hmlKmeansUpdateCtrdGmem<15, 15, 0, false>;
  kernel[16] = hmlKmeansUpdateCtrdGmem<16, 16, 0, false>;
  kernel[17] = hmlKmeansUpdateCtrdGmem<17, 17, 0, false>;
  kernel[18] = hmlKmeansUpdateCtrdGmem<18, 18, 0, false>;
  kernel[19] = hmlKmeansUpdateCtrdGmem<19, 19, 0, false>;
  kernel[20] = hmlKmeansUpdateCtrdGmem<20, 20, 0, false>;
  kernel[21] = hmlKmeansUpdateCtrdGmem<21, 21, 0, false>;
  kernel[22] = hmlKmeansUpdateCtrdGmem<22, 22, 0, false>;
  kernel[23] = hmlKmeansUpdateCtrdGmem<23, 23, 0, false>;
  kernel[24] = hmlKmeansUpdateCtrdGmem<24, 24, 0, false>;
  kernel[25] = hmlKmeansUpdateCtrdGmem<25, 25, 0, false>;
  kernel[26] = hmlKmeansUpdateCtrdGmem<26, 26, 0, false>;
  kernel[27] = hmlKmeansUpdateCtrdGmem<27, 27, 0, false>;
  kernel[28] = hmlKmeansUpdateCtrdGmem<28, 28, 0, false>;
  kernel[29] = hmlKmeansUpdateCtrdGmem<29, 29, 0, false>;
  kernel[30] = hmlKmeansUpdateCtrdGmem<30, 30, 0, false>;
  kernel[31] = hmlKmeansUpdateCtrdGmem<31, 31, 0, false>;
  kernel[32] = hmlKmeansUpdateCtrdGmem<32, 32, 0, false>;
  kernel[33] = hmlKmeansUpdateCtrdGmem<33, 32, 1, false>;
  kernel[34] = hmlKmeansUpdateCtrdGmem<34, 32, 2, false>;
  kernel[35] = hmlKmeansUpdateCtrdGmem<35, 32, 3, false>;
  kernel[36] = hmlKmeansUpdateCtrdGmem<36, 32, 4, false>;
  kernel[37] = hmlKmeansUpdateCtrdGmem<37, 32, 5, false>;
  kernel[38] = hmlKmeansUpdateCtrdGmem<38, 32, 6, false>;
  kernel[39] = hmlKmeansUpdateCtrdGmem<39, 32, 7, false>;
  kernel[40] = hmlKmeansUpdateCtrdGmem<40, 32, 8, false>;
  /*
  kernel[41] = hmlKmeansUpdateCtrdGmem<41, 32, 9, false>;
  kernel[42] = hmlKmeansUpdateCtrdGmem<42, 32, 10, false>;
  kernel[43] = hmlKmeansUpdateCtrdGmem<43, 32, 11, false>;
  kernel[44] = hmlKmeansUpdateCtrdGmem<44, 32, 12, false>;
  kernel[45] = hmlKmeansUpdateCtrdGmem<45, 32, 13, false>;
  kernel[46] = hmlKmeansUpdateCtrdGmem<46, 32, 14, false>;
  kernel[47] = hmlKmeansUpdateCtrdGmem<47, 32, 15, false>;
  kernel[48] = hmlKmeansUpdateCtrdGmem<48, 32, 16, false>;
  kernel[49] = hmlKmeansUpdateCtrdGmem<49, 32, 17, false>;
  kernel[50] = hmlKmeansUpdateCtrdGmem<50, 32, 18, false>;
  kernel[51] = hmlKmeansUpdateCtrdGmem<51, 32, 19, false>;
  kernel[52] = hmlKmeansUpdateCtrdGmem<52, 32, 20, false>;
  kernel[53] = hmlKmeansUpdateCtrdGmem<53, 32, 21, false>;
  kernel[54] = hmlKmeansUpdateCtrdGmem<54, 32, 22, false>;
  kernel[55] = hmlKmeansUpdateCtrdGmem<55, 32, 23, false>;
  kernel[56] = hmlKmeansUpdateCtrdGmem<56, 32, 24, false>;
  kernel[57] = hmlKmeansUpdateCtrdGmem<57, 32, 25, false>;
  kernel[58] = hmlKmeansUpdateCtrdGmem<58, 32, 26, false>;
  kernel[59] = hmlKmeansUpdateCtrdGmem<59, 32, 27, false>;
  kernel[60] = hmlKmeansUpdateCtrdGmem<60, 32, 28, false>;
  kernel[61] = hmlKmeansUpdateCtrdGmem<61, 32, 29, false>;
  kernel[62] = hmlKmeansUpdateCtrdGmem<62, 32, 30, false>;
  kernel[63] = hmlKmeansUpdateCtrdGmem<63, 32, 31, false>;
  kernel[64] = hmlKmeansUpdateCtrdGmem<64, 32, 32, false>;
  */
}

void
hmlKmeansInitKernelUpdateSmem(HmlKmeansUpdateKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansUpdateKernel) * (cHmlKmeansMaxDims + 1));

  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansUpdateCtrdSmem<1, 1, 0, false>; */
  kernel[2] = hmlKmeansUpdateCtrdSmem<2, 2, 0, false>;
  kernel[3] = hmlKmeansUpdateCtrdSmem<3, 3, 0, false>;
  kernel[4] = hmlKmeansUpdateCtrdSmem<4, 4, 0, false>;
  kernel[5] = hmlKmeansUpdateCtrdSmem<5, 5, 0, false>;
  kernel[6] = hmlKmeansUpdateCtrdSmem<6, 6, 0, false>;
  kernel[7] = hmlKmeansUpdateCtrdSmem<7, 7, 0, false>;
  kernel[8] = hmlKmeansUpdateCtrdSmem<8, 8, 0, false>;
  kernel[9] = hmlKmeansUpdateCtrdSmem<9, 9, 0, false>;
  kernel[10] = hmlKmeansUpdateCtrdSmem<10, 10, 0, false>;
  kernel[11] = hmlKmeansUpdateCtrdSmem<11, 11, 0, false>;
  kernel[12] = hmlKmeansUpdateCtrdSmem<12, 12, 0, false>;
  kernel[13] = hmlKmeansUpdateCtrdSmem<13, 13, 0, false>;
  kernel[14] = hmlKmeansUpdateCtrdSmem<14, 14, 0, false>;
  kernel[15] = hmlKmeansUpdateCtrdSmem<15, 15, 0, false>;
  kernel[16] = hmlKmeansUpdateCtrdSmem<16, 16, 0, false>;
  kernel[17] = hmlKmeansUpdateCtrdSmem<17, 17, 0, false>;
  kernel[18] = hmlKmeansUpdateCtrdSmem<18, 18, 0, false>;
  kernel[19] = hmlKmeansUpdateCtrdSmem<19, 19, 0, false>;
  kernel[20] = hmlKmeansUpdateCtrdSmem<20, 20, 0, false>;
  kernel[21] = hmlKmeansUpdateCtrdSmem<21, 21, 0, false>;
  kernel[22] = hmlKmeansUpdateCtrdSmem<22, 22, 0, false>;
  kernel[23] = hmlKmeansUpdateCtrdSmem<23, 23, 0, false>;
  kernel[24] = hmlKmeansUpdateCtrdSmem<24, 24, 0, false>;
  kernel[25] = hmlKmeansUpdateCtrdSmem<25, 25, 0, false>;
  kernel[26] = hmlKmeansUpdateCtrdSmem<26, 26, 0, false>;
  kernel[27] = hmlKmeansUpdateCtrdSmem<27, 27, 0, false>;
  kernel[28] = hmlKmeansUpdateCtrdSmem<28, 28, 0, false>;
  kernel[29] = hmlKmeansUpdateCtrdSmem<29, 29, 0, false>;
  kernel[30] = hmlKmeansUpdateCtrdSmem<30, 30, 0, false>;
  kernel[31] = hmlKmeansUpdateCtrdSmem<31, 31, 0, false>;
  kernel[32] = hmlKmeansUpdateCtrdSmem<32, 32, 0, false>;
  kernel[33] = hmlKmeansUpdateCtrdSmem<33, 32, 1, false>;
  kernel[34] = hmlKmeansUpdateCtrdSmem<34, 32, 2, false>;
  kernel[35] = hmlKmeansUpdateCtrdSmem<35, 32, 3, false>;
  kernel[36] = hmlKmeansUpdateCtrdSmem<36, 32, 4, false>;
  kernel[37] = hmlKmeansUpdateCtrdSmem<37, 32, 5, false>;
  kernel[38] = hmlKmeansUpdateCtrdSmem<38, 32, 6, false>;
  kernel[39] = hmlKmeansUpdateCtrdSmem<39, 32, 7, false>;
  kernel[40] = hmlKmeansUpdateCtrdSmem<40, 32, 8, false>;
  /*
  kernel[41] = hmlKmeansUpdateCtrdSmem<41, 32, 9, false>;
  kernel[42] = hmlKmeansUpdateCtrdSmem<42, 32, 10, false>;
  kernel[43] = hmlKmeansUpdateCtrdSmem<43, 32, 11, false>;
  kernel[44] = hmlKmeansUpdateCtrdSmem<44, 32, 12, false>;
  kernel[45] = hmlKmeansUpdateCtrdSmem<45, 32, 13, false>;
  kernel[46] = hmlKmeansUpdateCtrdSmem<46, 32, 14, false>;
  kernel[47] = hmlKmeansUpdateCtrdSmem<47, 32, 15, false>;
  kernel[48] = hmlKmeansUpdateCtrdSmem<48, 32, 16, false>;
  kernel[49] = hmlKmeansUpdateCtrdSmem<49, 32, 17, false>;
  kernel[50] = hmlKmeansUpdateCtrdSmem<50, 32, 18, false>;
  kernel[51] = hmlKmeansUpdateCtrdSmem<51, 32, 19, false>;
  kernel[52] = hmlKmeansUpdateCtrdSmem<52, 32, 20, false>;
  kernel[53] = hmlKmeansUpdateCtrdSmem<53, 32, 21, false>;
  kernel[54] = hmlKmeansUpdateCtrdSmem<54, 32, 22, false>;
  kernel[55] = hmlKmeansUpdateCtrdSmem<55, 32, 23, false>;
  kernel[56] = hmlKmeansUpdateCtrdSmem<56, 32, 24, false>;
  kernel[57] = hmlKmeansUpdateCtrdSmem<57, 32, 25, false>;
  kernel[58] = hmlKmeansUpdateCtrdSmem<58, 32, 26, false>;
  kernel[59] = hmlKmeansUpdateCtrdSmem<59, 32, 27, false>;
  kernel[60] = hmlKmeansUpdateCtrdSmem<60, 32, 28, false>;
  kernel[61] = hmlKmeansUpdateCtrdSmem<61, 32, 29, false>;
  kernel[62] = hmlKmeansUpdateCtrdSmem<62, 32, 30, false>;
  kernel[63] = hmlKmeansUpdateCtrdSmem<63, 32, 31, false>;
  kernel[64] = hmlKmeansUpdateCtrdSmem<64, 32, 32, false>;
  */
}

/* init kernels that use texture memory */
void
hmlKmeansInitKernelAssignGmemTex(HmlKmeansAssignKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansAssignKernel) * (cHmlKmeansMaxDims + 1));
  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansAssignCtrdGmem<1, true>; */
  kernel[2] = hmlKmeansAssignCtrdGmem<2, true>;
  kernel[3] = hmlKmeansAssignCtrdGmem<3, true>;
  kernel[4] = hmlKmeansAssignCtrdGmem<4, true>;
  kernel[5] = hmlKmeansAssignCtrdGmem<5, true>;
  kernel[6] = hmlKmeansAssignCtrdGmem<6, true>;
  kernel[7] = hmlKmeansAssignCtrdGmem<7, true>;
  kernel[8] = hmlKmeansAssignCtrdGmem<8, true>;
  kernel[9] = hmlKmeansAssignCtrdGmem<9, true>;
  kernel[10] = hmlKmeansAssignCtrdGmem<10, true>;
  kernel[11] = hmlKmeansAssignCtrdGmem<11, true>;
  kernel[12] = hmlKmeansAssignCtrdGmem<12, true>;
  kernel[13] = hmlKmeansAssignCtrdGmem<13, true>;
  kernel[14] = hmlKmeansAssignCtrdGmem<14, true>;
  kernel[15] = hmlKmeansAssignCtrdGmem<15, true>;
  kernel[16] = hmlKmeansAssignCtrdGmem<16, true>;
  kernel[17] = hmlKmeansAssignCtrdGmem<17, true>;
  kernel[18] = hmlKmeansAssignCtrdGmem<18, true>;
  kernel[19] = hmlKmeansAssignCtrdGmem<19, true>;
  kernel[20] = hmlKmeansAssignCtrdGmem<20, true>;
  kernel[21] = hmlKmeansAssignCtrdGmem<21, true>;
  kernel[22] = hmlKmeansAssignCtrdGmem<22, true>;
  kernel[23] = hmlKmeansAssignCtrdGmem<23, true>;
  kernel[24] = hmlKmeansAssignCtrdGmem<24, true>;
  kernel[25] = hmlKmeansAssignCtrdGmem<25, true>;
  kernel[26] = hmlKmeansAssignCtrdGmem<26, true>;
  kernel[27] = hmlKmeansAssignCtrdGmem<27, true>;
  kernel[28] = hmlKmeansAssignCtrdGmem<28, true>;
  kernel[29] = hmlKmeansAssignCtrdGmem<29, true>;
  kernel[30] = hmlKmeansAssignCtrdGmem<30, true>;
  kernel[31] = hmlKmeansAssignCtrdGmem<31, true>;
  kernel[32] = hmlKmeansAssignCtrdGmem<32, true>;
  kernel[33] = hmlKmeansAssignCtrdGmem<33, true>;
  kernel[34] = hmlKmeansAssignCtrdGmem<34, true>;
  kernel[35] = hmlKmeansAssignCtrdGmem<35, true>;
  kernel[36] = hmlKmeansAssignCtrdGmem<36, true>;
  kernel[37] = hmlKmeansAssignCtrdGmem<37, true>;
  kernel[38] = hmlKmeansAssignCtrdGmem<38, true>;
  kernel[39] = hmlKmeansAssignCtrdGmem<39, true>;
  kernel[40] = hmlKmeansAssignCtrdGmem<40, true>;
  /*
  kernel[41] = hmlKmeansAssignCtrdGmem<41, true>;
  kernel[42] = hmlKmeansAssignCtrdGmem<42, true>;
  kernel[43] = hmlKmeansAssignCtrdGmem<43, true>;
  kernel[44] = hmlKmeansAssignCtrdGmem<44, true>;
  kernel[45] = hmlKmeansAssignCtrdGmem<45, true>;
  kernel[46] = hmlKmeansAssignCtrdGmem<46, true>;
  kernel[47] = hmlKmeansAssignCtrdGmem<47, true>;
  kernel[48] = hmlKmeansAssignCtrdGmem<48, true>;
  kernel[49] = hmlKmeansAssignCtrdGmem<49, true>;
  kernel[50] = hmlKmeansAssignCtrdGmem<50, true>;
  kernel[51] = hmlKmeansAssignCtrdGmem<51, true>;
  kernel[52] = hmlKmeansAssignCtrdGmem<52, true>;
  kernel[53] = hmlKmeansAssignCtrdGmem<53, true>;
  kernel[54] = hmlKmeansAssignCtrdGmem<54, true>;
  kernel[55] = hmlKmeansAssignCtrdGmem<55, true>;
  kernel[56] = hmlKmeansAssignCtrdGmem<56, true>;
  kernel[57] = hmlKmeansAssignCtrdGmem<57, true>;
  kernel[58] = hmlKmeansAssignCtrdGmem<58, true>;
  kernel[59] = hmlKmeansAssignCtrdGmem<59, true>;
  kernel[60] = hmlKmeansAssignCtrdGmem<60, true>;
  kernel[61] = hmlKmeansAssignCtrdGmem<61, true>;
  kernel[62] = hmlKmeansAssignCtrdGmem<62, true>;
  kernel[63] = hmlKmeansAssignCtrdGmem<63, true>;
  kernel[64] = hmlKmeansAssignCtrdGmem<64, true>;
  */
}

void
hmlKmeansInitKernelAssignSmemTex(HmlKmeansAssignKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansAssignKernel) * (cHmlKmeansMaxDims + 1));

  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansAssignCtrdSmem<1, true>; */
  kernel[2] = hmlKmeansAssignCtrdSmem<2, true>;
  kernel[3] = hmlKmeansAssignCtrdSmem<3, true>;
  kernel[4] = hmlKmeansAssignCtrdSmem<4, true>;
  kernel[5] = hmlKmeansAssignCtrdSmem<5, true>;
  kernel[6] = hmlKmeansAssignCtrdSmem<6, true>;
  kernel[7] = hmlKmeansAssignCtrdSmem<7, true>;
  kernel[8] = hmlKmeansAssignCtrdSmem<8, true>;
  kernel[9] = hmlKmeansAssignCtrdSmem<9, true>;
  kernel[10] = hmlKmeansAssignCtrdSmem<10, true>;
  kernel[11] = hmlKmeansAssignCtrdSmem<11, true>;
  kernel[12] = hmlKmeansAssignCtrdSmem<12, true>;
  kernel[13] = hmlKmeansAssignCtrdSmem<13, true>;
  kernel[14] = hmlKmeansAssignCtrdSmem<14, true>;
  kernel[15] = hmlKmeansAssignCtrdSmem<15, true>;
  kernel[16] = hmlKmeansAssignCtrdSmem<16, true>;
  kernel[17] = hmlKmeansAssignCtrdSmem<17, true>;
  kernel[18] = hmlKmeansAssignCtrdSmem<18, true>;
  kernel[19] = hmlKmeansAssignCtrdSmem<19, true>;
  kernel[20] = hmlKmeansAssignCtrdSmem<20, true>;
  kernel[21] = hmlKmeansAssignCtrdSmem<21, true>;
  kernel[22] = hmlKmeansAssignCtrdSmem<22, true>;
  kernel[23] = hmlKmeansAssignCtrdSmem<23, true>;
  kernel[24] = hmlKmeansAssignCtrdSmem<24, true>;
  kernel[25] = hmlKmeansAssignCtrdSmem<25, true>;
  kernel[26] = hmlKmeansAssignCtrdSmem<26, true>;
  kernel[27] = hmlKmeansAssignCtrdSmem<27, true>;
  kernel[28] = hmlKmeansAssignCtrdSmem<28, true>;
  kernel[29] = hmlKmeansAssignCtrdSmem<29, true>;
  kernel[30] = hmlKmeansAssignCtrdSmem<30, true>;
  kernel[31] = hmlKmeansAssignCtrdSmem<31, true>;
  kernel[32] = hmlKmeansAssignCtrdSmem<32, true>;
  kernel[33] = hmlKmeansAssignCtrdSmem<33, true>;
  kernel[34] = hmlKmeansAssignCtrdSmem<34, true>;
  kernel[35] = hmlKmeansAssignCtrdSmem<35, true>;
  kernel[36] = hmlKmeansAssignCtrdSmem<36, true>;
  kernel[37] = hmlKmeansAssignCtrdSmem<37, true>;
  kernel[38] = hmlKmeansAssignCtrdSmem<38, true>;
  kernel[39] = hmlKmeansAssignCtrdSmem<39, true>;
  kernel[40] = hmlKmeansAssignCtrdSmem<40, true>;
  /*
  kernel[41] = hmlKmeansAssignCtrdSmem<41, true>;
  kernel[42] = hmlKmeansAssignCtrdSmem<42, true>;
  kernel[43] = hmlKmeansAssignCtrdSmem<43, true>;
  kernel[44] = hmlKmeansAssignCtrdSmem<44, true>;
  kernel[45] = hmlKmeansAssignCtrdSmem<45, true>;
  kernel[46] = hmlKmeansAssignCtrdSmem<46, true>;
  kernel[47] = hmlKmeansAssignCtrdSmem<47, true>;
  kernel[48] = hmlKmeansAssignCtrdSmem<48, true>;
  kernel[49] = hmlKmeansAssignCtrdSmem<49, true>;
  kernel[50] = hmlKmeansAssignCtrdSmem<50, true>;
  kernel[51] = hmlKmeansAssignCtrdSmem<51, true>;
  kernel[52] = hmlKmeansAssignCtrdSmem<52, true>;
  kernel[53] = hmlKmeansAssignCtrdSmem<53, true>;
  kernel[54] = hmlKmeansAssignCtrdSmem<54, true>;
  kernel[55] = hmlKmeansAssignCtrdSmem<55, true>;
  kernel[56] = hmlKmeansAssignCtrdSmem<56, true>;
  kernel[57] = hmlKmeansAssignCtrdSmem<57, true>;
  kernel[58] = hmlKmeansAssignCtrdSmem<58, true>;
  kernel[59] = hmlKmeansAssignCtrdSmem<59, true>;
  kernel[60] = hmlKmeansAssignCtrdSmem<60, true>;
  kernel[61] = hmlKmeansAssignCtrdSmem<61, true>;
  kernel[62] = hmlKmeansAssignCtrdSmem<62, true>;
  kernel[63] = hmlKmeansAssignCtrdSmem<63, true>;
  kernel[64] = hmlKmeansAssignCtrdSmem<64, true>;
  */
}

void
hmlKmeansInitKernelAssignSmemUnrollTex(HmlKmeansAssignKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansAssignKernel) * (cHmlKmeansMaxDims + 1));

  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansAssignCtrdSmemUnroll<1, true>; */
  kernel[2] = hmlKmeansAssignCtrdSmemUnroll<2, true>;
  kernel[3] = hmlKmeansAssignCtrdSmemUnroll<3, true>;
  kernel[4] = hmlKmeansAssignCtrdSmemUnroll<4, true>;
  kernel[5] = hmlKmeansAssignCtrdSmemUnroll<5, true>;
  kernel[6] = hmlKmeansAssignCtrdSmemUnroll<6, true>;
  kernel[7] = hmlKmeansAssignCtrdSmemUnroll<7, true>;
  kernel[8] = hmlKmeansAssignCtrdSmemUnroll<8, true>;
  kernel[9] = hmlKmeansAssignCtrdSmemUnroll<9, true>;
  kernel[10] = hmlKmeansAssignCtrdSmemUnroll<10, true>;
  kernel[11] = hmlKmeansAssignCtrdSmemUnroll<11, true>;
  kernel[12] = hmlKmeansAssignCtrdSmemUnroll<12, true>;
  kernel[13] = hmlKmeansAssignCtrdSmemUnroll<13, true>;
  kernel[14] = hmlKmeansAssignCtrdSmemUnroll<14, true>;
  kernel[15] = hmlKmeansAssignCtrdSmemUnroll<15, true>;
  kernel[16] = hmlKmeansAssignCtrdSmemUnroll<16, true>;
  kernel[17] = hmlKmeansAssignCtrdSmemUnroll<17, true>;
  kernel[18] = hmlKmeansAssignCtrdSmemUnroll<18, true>;
  kernel[19] = hmlKmeansAssignCtrdSmemUnroll<19, true>;
  kernel[20] = hmlKmeansAssignCtrdSmemUnroll<20, true>;
  kernel[21] = hmlKmeansAssignCtrdSmemUnroll<21, true>;
  kernel[22] = hmlKmeansAssignCtrdSmemUnroll<22, true>;
  kernel[23] = hmlKmeansAssignCtrdSmemUnroll<23, true>;
  kernel[24] = hmlKmeansAssignCtrdSmemUnroll<24, true>;
  kernel[25] = hmlKmeansAssignCtrdSmemUnroll<25, true>;
  kernel[26] = hmlKmeansAssignCtrdSmemUnroll<26, true>;
  kernel[27] = hmlKmeansAssignCtrdSmemUnroll<27, true>;
  kernel[28] = hmlKmeansAssignCtrdSmemUnroll<28, true>;
  kernel[29] = hmlKmeansAssignCtrdSmemUnroll<29, true>;
  kernel[30] = hmlKmeansAssignCtrdSmemUnroll<30, true>;
  kernel[31] = hmlKmeansAssignCtrdSmemUnroll<31, true>;
  kernel[32] = hmlKmeansAssignCtrdSmemUnroll<32, true>;
  kernel[33] = hmlKmeansAssignCtrdSmemUnroll<33, true>;
  kernel[34] = hmlKmeansAssignCtrdSmemUnroll<34, true>;
  kernel[35] = hmlKmeansAssignCtrdSmemUnroll<35, true>;
  kernel[36] = hmlKmeansAssignCtrdSmemUnroll<36, true>;
  kernel[37] = hmlKmeansAssignCtrdSmemUnroll<37, true>;
  kernel[38] = hmlKmeansAssignCtrdSmemUnroll<38, true>;
  kernel[39] = hmlKmeansAssignCtrdSmemUnroll<39, true>;
  kernel[40] = hmlKmeansAssignCtrdSmemUnroll<40, true>;
  /*
  kernel[41] = hmlKmeansAssignCtrdSmemUnroll<41, true>;
  kernel[42] = hmlKmeansAssignCtrdSmemUnroll<42, true>;
  kernel[43] = hmlKmeansAssignCtrdSmemUnroll<43, true>;
  kernel[44] = hmlKmeansAssignCtrdSmemUnroll<44, true>;
  kernel[45] = hmlKmeansAssignCtrdSmemUnroll<45, true>;
  kernel[46] = hmlKmeansAssignCtrdSmemUnroll<46, true>;
  kernel[47] = hmlKmeansAssignCtrdSmemUnroll<47, true>;
  kernel[48] = hmlKmeansAssignCtrdSmemUnroll<48, true>;
  kernel[49] = hmlKmeansAssignCtrdSmemUnroll<49, true>;
  kernel[50] = hmlKmeansAssignCtrdSmemUnroll<50, true>;
  kernel[51] = hmlKmeansAssignCtrdSmemUnroll<51, true>;
  kernel[52] = hmlKmeansAssignCtrdSmemUnroll<52, true>;
  kernel[53] = hmlKmeansAssignCtrdSmemUnroll<53, true>;
  kernel[54] = hmlKmeansAssignCtrdSmemUnroll<54, true>;
  kernel[55] = hmlKmeansAssignCtrdSmemUnroll<55, true>;
  kernel[56] = hmlKmeansAssignCtrdSmemUnroll<56, true>;
  kernel[57] = hmlKmeansAssignCtrdSmemUnroll<57, true>;
  kernel[58] = hmlKmeansAssignCtrdSmemUnroll<58, true>;
  kernel[59] = hmlKmeansAssignCtrdSmemUnroll<59, true>;
  kernel[60] = hmlKmeansAssignCtrdSmemUnroll<60, true>;
  kernel[61] = hmlKmeansAssignCtrdSmemUnroll<61, true>;
  kernel[62] = hmlKmeansAssignCtrdSmemUnroll<62, true>;
  kernel[63] = hmlKmeansAssignCtrdSmemUnroll<63, true>;
  kernel[64] = hmlKmeansAssignCtrdSmemUnroll<64, true>;
  */
}

void
hmlKmeansInitKernelUpdateGmemTex(HmlKmeansUpdateKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansUpdateKernel) * (cHmlKmeansMaxDims + 1));

  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansUpdateCtrdGmem<1, 1, 0, true>; */
  kernel[2] = hmlKmeansUpdateCtrdGmem<2, 2, 0, true>;
  kernel[3] = hmlKmeansUpdateCtrdGmem<3, 3, 0, true>;
  kernel[4] = hmlKmeansUpdateCtrdGmem<4, 4, 0, true>;
  kernel[5] = hmlKmeansUpdateCtrdGmem<5, 5, 0, true>;
  kernel[6] = hmlKmeansUpdateCtrdGmem<6, 6, 0, true>;
  kernel[7] = hmlKmeansUpdateCtrdGmem<7, 7, 0, true>;
  kernel[8] = hmlKmeansUpdateCtrdGmem<8, 8, 0, true>;
  kernel[9] = hmlKmeansUpdateCtrdGmem<9, 9, 0, true>;
  kernel[10] = hmlKmeansUpdateCtrdGmem<10, 10, 0, true>;
  kernel[11] = hmlKmeansUpdateCtrdGmem<11, 11, 0, true>;
  kernel[12] = hmlKmeansUpdateCtrdGmem<12, 12, 0, true>;
  kernel[13] = hmlKmeansUpdateCtrdGmem<13, 13, 0, true>;
  kernel[14] = hmlKmeansUpdateCtrdGmem<14, 14, 0, true>;
  kernel[15] = hmlKmeansUpdateCtrdGmem<15, 15, 0, true>;
  kernel[16] = hmlKmeansUpdateCtrdGmem<16, 16, 0, true>;
  kernel[17] = hmlKmeansUpdateCtrdGmem<17, 17, 0, true>;
  kernel[18] = hmlKmeansUpdateCtrdGmem<18, 18, 0, true>;
  kernel[19] = hmlKmeansUpdateCtrdGmem<19, 19, 0, true>;
  kernel[20] = hmlKmeansUpdateCtrdGmem<20, 20, 0, true>;
  kernel[21] = hmlKmeansUpdateCtrdGmem<21, 21, 0, true>;
  kernel[22] = hmlKmeansUpdateCtrdGmem<22, 22, 0, true>;
  kernel[23] = hmlKmeansUpdateCtrdGmem<23, 23, 0, true>;
  kernel[24] = hmlKmeansUpdateCtrdGmem<24, 24, 0, true>;
  kernel[25] = hmlKmeansUpdateCtrdGmem<25, 25, 0, true>;
  kernel[26] = hmlKmeansUpdateCtrdGmem<26, 26, 0, true>;
  kernel[27] = hmlKmeansUpdateCtrdGmem<27, 27, 0, true>;
  kernel[28] = hmlKmeansUpdateCtrdGmem<28, 28, 0, true>;
  kernel[29] = hmlKmeansUpdateCtrdGmem<29, 29, 0, true>;
  kernel[30] = hmlKmeansUpdateCtrdGmem<30, 30, 0, true>;
  kernel[31] = hmlKmeansUpdateCtrdGmem<31, 31, 0, true>;
  kernel[32] = hmlKmeansUpdateCtrdGmem<32, 32, 0, true>;
  kernel[33] = hmlKmeansUpdateCtrdGmem<33, 32, 1, true>;
  kernel[34] = hmlKmeansUpdateCtrdGmem<34, 32, 2, true>;
  kernel[35] = hmlKmeansUpdateCtrdGmem<35, 32, 3, true>;
  kernel[36] = hmlKmeansUpdateCtrdGmem<36, 32, 4, true>;
  kernel[37] = hmlKmeansUpdateCtrdGmem<37, 32, 5, true>;
  kernel[38] = hmlKmeansUpdateCtrdGmem<38, 32, 6, true>;
  kernel[39] = hmlKmeansUpdateCtrdGmem<39, 32, 7, true>;
  kernel[40] = hmlKmeansUpdateCtrdGmem<40, 32, 8, true>;
  /*
  kernel[41] = hmlKmeansUpdateCtrdGmem<41, 32, 9, true>;
  kernel[42] = hmlKmeansUpdateCtrdGmem<42, 32, 10, true>;
  kernel[43] = hmlKmeansUpdateCtrdGmem<43, 32, 11, true>;
  kernel[44] = hmlKmeansUpdateCtrdGmem<44, 32, 12, true>;
  kernel[45] = hmlKmeansUpdateCtrdGmem<45, 32, 13, true>;
  kernel[46] = hmlKmeansUpdateCtrdGmem<46, 32, 14, true>;
  kernel[47] = hmlKmeansUpdateCtrdGmem<47, 32, 15, true>;
  kernel[48] = hmlKmeansUpdateCtrdGmem<48, 32, 16, true>;
  kernel[49] = hmlKmeansUpdateCtrdGmem<49, 32, 17, true>;
  kernel[50] = hmlKmeansUpdateCtrdGmem<50, 32, 18, true>;
  kernel[51] = hmlKmeansUpdateCtrdGmem<51, 32, 19, true>;
  kernel[52] = hmlKmeansUpdateCtrdGmem<52, 32, 20, true>;
  kernel[53] = hmlKmeansUpdateCtrdGmem<53, 32, 21, true>;
  kernel[54] = hmlKmeansUpdateCtrdGmem<54, 32, 22, true>;
  kernel[55] = hmlKmeansUpdateCtrdGmem<55, 32, 23, true>;
  kernel[56] = hmlKmeansUpdateCtrdGmem<56, 32, 24, true>;
  kernel[57] = hmlKmeansUpdateCtrdGmem<57, 32, 25, true>;
  kernel[58] = hmlKmeansUpdateCtrdGmem<58, 32, 26, true>;
  kernel[59] = hmlKmeansUpdateCtrdGmem<59, 32, 27, true>;
  kernel[60] = hmlKmeansUpdateCtrdGmem<60, 32, 28, true>;
  kernel[61] = hmlKmeansUpdateCtrdGmem<61, 32, 29, true>;
  kernel[62] = hmlKmeansUpdateCtrdGmem<62, 32, 30, true>;
  kernel[63] = hmlKmeansUpdateCtrdGmem<63, 32, 31, true>;
  kernel[64] = hmlKmeansUpdateCtrdGmem<64, 32, 32, true>;
  */
}

void
hmlKmeansInitKernelUpdateSmemTex(HmlKmeansUpdateKernel *kernel)
{
  memset(kernel, 0, sizeof(HmlKmeansUpdateKernel) * (cHmlKmeansMaxDims + 1));

  /* The 1D kernel function is disabled so as to avoid the warning:
   * "Advisory: Loop was not unrolled, cannot deduce loop trip count"
   */
  /* kernel[1] = hmlKmeansUpdateCtrdSmem<1, 1, 0, true>; */
  kernel[2] = hmlKmeansUpdateCtrdSmem<2, 2, 0, true>;
  kernel[3] = hmlKmeansUpdateCtrdSmem<3, 3, 0, true>;
  kernel[4] = hmlKmeansUpdateCtrdSmem<4, 4, 0, true>;
  kernel[5] = hmlKmeansUpdateCtrdSmem<5, 5, 0, true>;
  kernel[6] = hmlKmeansUpdateCtrdSmem<6, 6, 0, true>;
  kernel[7] = hmlKmeansUpdateCtrdSmem<7, 7, 0, true>;
  kernel[8] = hmlKmeansUpdateCtrdSmem<8, 8, 0, true>;
  kernel[9] = hmlKmeansUpdateCtrdSmem<9, 9, 0, true>;
  kernel[10] = hmlKmeansUpdateCtrdSmem<10, 10, 0, true>;
  kernel[11] = hmlKmeansUpdateCtrdSmem<11, 11, 0, true>;
  kernel[12] = hmlKmeansUpdateCtrdSmem<12, 12, 0, true>;
  kernel[13] = hmlKmeansUpdateCtrdSmem<13, 13, 0, true>;
  kernel[14] = hmlKmeansUpdateCtrdSmem<14, 14, 0, true>;
  kernel[15] = hmlKmeansUpdateCtrdSmem<15, 15, 0, true>;
  kernel[16] = hmlKmeansUpdateCtrdSmem<16, 16, 0, true>;
  kernel[17] = hmlKmeansUpdateCtrdSmem<17, 17, 0, true>;
  kernel[18] = hmlKmeansUpdateCtrdSmem<18, 18, 0, true>;
  kernel[19] = hmlKmeansUpdateCtrdSmem<19, 19, 0, true>;
  kernel[20] = hmlKmeansUpdateCtrdSmem<20, 20, 0, true>;
  kernel[21] = hmlKmeansUpdateCtrdSmem<21, 21, 0, true>;
  kernel[22] = hmlKmeansUpdateCtrdSmem<22, 22, 0, true>;
  kernel[23] = hmlKmeansUpdateCtrdSmem<23, 23, 0, true>;
  kernel[24] = hmlKmeansUpdateCtrdSmem<24, 24, 0, true>;
  kernel[25] = hmlKmeansUpdateCtrdSmem<25, 25, 0, true>;
  kernel[26] = hmlKmeansUpdateCtrdSmem<26, 26, 0, true>;
  kernel[27] = hmlKmeansUpdateCtrdSmem<27, 27, 0, true>;
  kernel[28] = hmlKmeansUpdateCtrdSmem<28, 28, 0, true>;
  kernel[29] = hmlKmeansUpdateCtrdSmem<29, 29, 0, true>;
  kernel[30] = hmlKmeansUpdateCtrdSmem<30, 30, 0, true>;
  kernel[31] = hmlKmeansUpdateCtrdSmem<31, 31, 0, true>;
  kernel[32] = hmlKmeansUpdateCtrdSmem<32, 32, 0, true>;
  kernel[33] = hmlKmeansUpdateCtrdSmem<33, 32, 1, true>;
  kernel[34] = hmlKmeansUpdateCtrdSmem<34, 32, 2, true>;
  kernel[35] = hmlKmeansUpdateCtrdSmem<35, 32, 3, true>;
  kernel[36] = hmlKmeansUpdateCtrdSmem<36, 32, 4, true>;
  kernel[37] = hmlKmeansUpdateCtrdSmem<37, 32, 5, true>;
  kernel[38] = hmlKmeansUpdateCtrdSmem<38, 32, 6, true>;
  kernel[39] = hmlKmeansUpdateCtrdSmem<39, 32, 7, true>;
  kernel[40] = hmlKmeansUpdateCtrdSmem<40, 32, 8, true>;
  /*
  kernel[41] = hmlKmeansUpdateCtrdSmem<41, 32, 9, true>;
  kernel[42] = hmlKmeansUpdateCtrdSmem<42, 32, 10, true>;
  kernel[43] = hmlKmeansUpdateCtrdSmem<43, 32, 11, true>;
  kernel[44] = hmlKmeansUpdateCtrdSmem<44, 32, 12, true>;
  kernel[45] = hmlKmeansUpdateCtrdSmem<45, 32, 13, true>;
  kernel[46] = hmlKmeansUpdateCtrdSmem<46, 32, 14, true>;
  kernel[47] = hmlKmeansUpdateCtrdSmem<47, 32, 15, true>;
  kernel[48] = hmlKmeansUpdateCtrdSmem<48, 32, 16, true>;
  kernel[49] = hmlKmeansUpdateCtrdSmem<49, 32, 17, true>;
  kernel[50] = hmlKmeansUpdateCtrdSmem<50, 32, 18, true>;
  kernel[51] = hmlKmeansUpdateCtrdSmem<51, 32, 19, true>;
  kernel[52] = hmlKmeansUpdateCtrdSmem<52, 32, 20, true>;
  kernel[53] = hmlKmeansUpdateCtrdSmem<53, 32, 21, true>;
  kernel[54] = hmlKmeansUpdateCtrdSmem<54, 32, 22, true>;
  kernel[55] = hmlKmeansUpdateCtrdSmem<55, 32, 23, true>;
  kernel[56] = hmlKmeansUpdateCtrdSmem<56, 32, 24, true>;
  kernel[57] = hmlKmeansUpdateCtrdSmem<57, 32, 25, true>;
  kernel[58] = hmlKmeansUpdateCtrdSmem<58, 32, 26, true>;
  kernel[59] = hmlKmeansUpdateCtrdSmem<59, 32, 27, true>;
  kernel[60] = hmlKmeansUpdateCtrdSmem<60, 32, 28, true>;
  kernel[61] = hmlKmeansUpdateCtrdSmem<61, 32, 29, true>;
  kernel[62] = hmlKmeansUpdateCtrdSmem<62, 32, 30, true>;
  kernel[63] = hmlKmeansUpdateCtrdSmem<63, 32, 31, true>;
  kernel[64] = hmlKmeansUpdateCtrdSmem<64, 32, 32, true>;
  */
}
