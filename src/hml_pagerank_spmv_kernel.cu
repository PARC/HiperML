/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_pagerank_spmv_kernel.h"
#include "hml_pagerank_spmv_kernel_template.h"
#include <assert.h>

/* constants limited by CUDA */
#define cHmlBlocksPerGrid      128
#define cHmlThreadsPerBlock    128

/* SpMV specific constants */
#define cHmlPagerankSpmvThreadsPerBlock    512

__global__ void
hmlPagerankSpmvInitKernel(float *devMap, float initVal, uint32_t numVertices)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < numVertices) {
    devMap[tid] = initVal;
    tid += blockDim.x * gridDim.x;
  }
}

void
hmlPagerankSpmvKernelArgPrint(HmlPagerankSpmvKernelArg *kernelArg, uint32_t numPartitions)
{
  for (uint32_t i = 0; i < numPartitions; ++i) {
    fprintf(stderr, "; Info: Kernel arg #%d:\n", i);
    fprintf(stderr, "; Info:     id            = %d\n", kernelArg[i].id);
    fprintf(stderr, "; Info:     minDeg        = %d\n", kernelArg[i].minDeg);
    fprintf(stderr, "; Info:     maxDeg        = %d\n", kernelArg[i].maxDeg);
    fprintf(stderr, "; Info:     minVertexRank = %d\n",
      kernelArg[i].minVertexRank);
    fprintf(stderr, "; Info:     maxVertexRank = %d\n",
      kernelArg[i].maxVertexRank);
    fprintf(stderr, "; Info:     # of vertices = %d\n",
            kernelArg[i].maxVertexRank - kernelArg[i].minVertexRank + 1);
  }
}

/* calculates the {x,y,1} size of the grid, provided that
 * gridDimXMin <= x <= gridDimXMax and
 * x <= cHmlMaxGridDimX and
 * y <= cHmlMaxGridDimY and
 * x * y ~= numBlocks
 */
void
hmlPagerankSpmvGridDimCalc(dim3   *grid,
            uint32_t  numBlocks,
            uint32_t  gridDimXMin,
            uint32_t  gridDimXMax)
{
  uint32_t gridDimX;
  uint32_t gridDimY;

  if (gridDimXMax > cHmlMaxGridDimX) {
    fprintf(stderr, "; Error: gridDimXMax = %d > cHmlMaxGridDimX = %d\n",
            gridDimXMax, cHmlMaxGridDimX);
    exit(EXIT_FAILURE);
  }
  /* double gridDimX until gridDimY is no more than cHmlMaxGridDimY */
  for (gridDimX = gridDimXMin; gridDimX <= gridDimXMax; gridDimX *= 2) {
    gridDimY = (numBlocks + gridDimX - 1) / gridDimX;
    if (gridDimY <= cHmlMaxGridDimY)
      break;
  }
  if (gridDimX > gridDimXMax) {
    fprintf(stderr, "; Error: gridDimX > gridDimXMax\n");
    exit(EXIT_FAILURE);
  }
  grid->x = gridDimX;
  grid->y = gridDimY;
  grid->z = 1;         /* always = 1 */
}

void
hmlPagerankSpmvKernelArgSet(HmlPagerankSpmvKernelArg *kernelArg,
             uint32_t     numPartitions,
             uint32_t    *minOutDeg, /* min out-degree of each partition */
             uint32_t    *vertexRank,
             uint32_t    *R,
             uint32_t    *partitionPrefixSize)
{
  uint32_t          p;
  uint32_t          numVertices;
  uint32_t          numBlocks;
  uint32_t          numThreadsPerBlock;

  for (p = 0; p < numPartitions; ++p) {
    kernelArg[p].minDeg = minOutDeg[p];
    if (p < numPartitions - 1)
      kernelArg[p].maxDeg = minOutDeg[p + 1];
    if (p > 0)
      kernelArg[p].minVertexRank = partitionPrefixSize[p - 1];
    else
      kernelArg[0].minVertexRank = 0;
    kernelArg[p].maxVertexRank = partitionPrefixSize[p];
  }
  /* the last kernel instance always has maxDeg = infinity */
  kernelArg[numPartitions - 1].maxDeg = (uint32_t)-1;

  /* setup the grid and block args for each partition */
  for (p = 0; p < numPartitions; ++p) {
    numVertices =
      kernelArg[p].maxVertexRank - kernelArg[p].minVertexRank; /* no +1 */
    switch (kernelArg[p].id) {
    case 0:
      kernelArg[p].block.x = 8;
      kernelArg[p].block.y = 8;
      kernelArg[p].block.z = 1;
      numThreadsPerBlock = kernelArg[p].block.x * kernelArg[p].block.y;
      numBlocks = (numVertices + numThreadsPerBlock - 1) / numThreadsPerBlock;
      /* it's OK to have gridDim.x be 1 for this kernel */
      hmlPagerankSpmvGridDimCalc(&kernelArg[p].grid, numBlocks, 1, cHmlMaxGridDimX);
      break;

    case 1:
      //kernelArg[p].block.x = MIN(kernelArg[p].minDeg, cHmlThreadsPerWarp);
      kernelArg[p].block.x = cHmlThreadsPerWarp;
      kernelArg[p].block.y = 1;
      kernelArg[p].block.z = 1;
      numBlocks = numVertices;
      /* gridDimX must >= 8, or CUDA behaves strangely */
      hmlPagerankSpmvGridDimCalc(&kernelArg[p].grid, numBlocks, 8, cHmlMaxGridDimX);
      break;

    case 2:
      //kernelArg[p].block.x = MIN(kernelArg[p].minDeg, cHmlPagerankSpmvThreadsPerBlock);
      kernelArg[p].block.x = cHmlPagerankSpmvThreadsPerBlock;
      kernelArg[p].block.y = 1;
      kernelArg[p].block.z = 1;
      numBlocks = numVertices;
      /* gridDimX must >= 8, or CUDA behaves strangely */
      hmlPagerankSpmvGridDimCalc(&kernelArg[p].grid, numBlocks, 8, cHmlMaxGridDimX);
      break;

    case 3:
      kernelArg[p].block.x = cHmlThreadsPerBlock;
      kernelArg[p].block.y = kernelArg[p].block.z = 1;
      kernelArg[p].grid.x = cHmlBlocksPerGrid;
      kernelArg[p].grid.y = kernelArg[p].grid.z = 1;
      break;

    default:
      fprintf(stderr, "; Error: Unknown kernel id: %d\n", kernelArg[p].id);
      exit(EXIT_FAILURE);
    }
  }
}

void
hmlPagerankSpmvKernelSetup(PagerankSpmvKernel       *evenIterKernel,
                           PagerankSpmvKernel       *oddIterKernel,
                           uint32_t               *minOutDegArr,
                           uint32_t                maxNumKernels,
                           bool                  useTextureMem,
                           uint32_t               *numPartitions,
                           HmlPagerankSpmvKernelArg *kernelArgArr)
{
  assert(maxNumKernels >= 3);

  *numPartitions = 3;
  evenIterKernel[0] = useTextureMem ?
    hmlPagerankSpmvKernelEvenIter0<true> : hmlPagerankSpmvKernelEvenIter0<false>;
  evenIterKernel[1] = useTextureMem ?
    hmlPagerankSpmvKernelEvenIter1<true> : hmlPagerankSpmvKernelEvenIter1<false>;
  evenIterKernel[2] = useTextureMem ?
    hmlPagerankSpmvKernelEvenIter2<true> : hmlPagerankSpmvKernelEvenIter2<false>;

  oddIterKernel[0] = useTextureMem ?
    hmlPagerankSpmvKernelOddIter0<true> : hmlPagerankSpmvKernelOddIter0<false>;
  oddIterKernel[1] = useTextureMem ?
    hmlPagerankSpmvKernelOddIter1<true> : hmlPagerankSpmvKernelOddIter1<false>;
  oddIterKernel[2] = useTextureMem ?
    hmlPagerankSpmvKernelOddIter2<true> : hmlPagerankSpmvKernelOddIter2<false>;

  /* set CUDA cache preference for kernels
   * cudaFuncCachePreferL1 makes code run ~6% faster
   */
  HANDLE_ERROR(cudaFuncSetCacheConfig(evenIterKernel[0], cudaFuncCachePreferL1));
  HANDLE_ERROR(cudaFuncSetCacheConfig(evenIterKernel[1], cudaFuncCachePreferL1));
  HANDLE_ERROR(cudaFuncSetCacheConfig(evenIterKernel[2], cudaFuncCachePreferL1));

  HANDLE_ERROR(cudaFuncSetCacheConfig(oddIterKernel[0], cudaFuncCachePreferL1));
  HANDLE_ERROR(cudaFuncSetCacheConfig(oddIterKernel[1], cudaFuncCachePreferL1));
  HANDLE_ERROR(cudaFuncSetCacheConfig(oddIterKernel[2], cudaFuncCachePreferL1));

  /* setup minOutDeg array */
  minOutDegArr[0] = 1;
  minOutDegArr[1] = 32;
  minOutDegArr[2] = 1024;

  /* init kernel arg array */
  kernelArgArr[0].id = 0; /* use kernel id 0 for first partition */
  kernelArgArr[1].id = 1; /* use kernel id 1 for second partition */
  kernelArgArr[2].id = 2; /* use kernel id 2 for third partition */
}
