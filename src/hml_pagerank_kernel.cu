/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_pagerank_kernel.h"
#include "hml_pagerank_kernel_template.h"
#include <assert.h>

/* constants limited by CUDA */
#define cHmlBlocksPerGrid      128
#define cHmlThreadsPerBlock    128

/* SpMV specific constants */
#define cHmlPagerankThreadsPerBlock    512

__global__ void
hmlPagerankInitKernel(float *devMap, float initVal, uint32_t numVertices) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while(tid < numVertices) {
    devMap[tid] = initVal;
    tid += blockDim.x * gridDim.x;
  }
}

void
hmlPagerankKernelArgPrint(HmlPagerankKernelArg *kernelArgs, uint32_t numPartitions) {
  for(uint32_t i = 0; i < numPartitions; ++i) {
    fprintf(stderr, "; Info: Kernel arg #%d:\n", i);
    fprintf(stderr, "; Info:     id            = %d\n", kernelArgs[i].id);
    fprintf(stderr, "; Info:     minDeg        = %d\n", kernelArgs[i].minDeg);
    fprintf(stderr, "; Info:     maxDeg        = %d\n", kernelArgs[i].maxDeg);
    fprintf(stderr, "; Info:     minVertexRank = %d\n",
            kernelArgs[i].minVertexRank);
    fprintf(stderr, "; Info:     maxVertexRank = %d\n",
            kernelArgs[i].maxVertexRank);
    fprintf(stderr, "; Info:     # of vertices = %d\n",
            kernelArgs[i].maxVertexRank - kernelArgs[i].minVertexRank + 1);
  }
}

/* calculates the {x,y,1} size of the grid, provided that
 * gridDimXMin <= x <= gridDimXMax and
 * x <= cHmlMaxGridDimX and
 * y <= cHmlMaxGridDimY and
 * x * y ~= numBlocks
 */
void
hmlPagerankGridDimCalc(dim3   *grid,
                       uint32_t  numBlocks,
                       uint32_t  gridDimXMin,
                       uint32_t  gridDimXMax) {
  uint32_t gridDimX;
  uint32_t gridDimY;

  if(gridDimXMax > cHmlMaxGridDimX) {
    fprintf(stderr, "; Error: gridDimXMax = %d > cHmlMaxGridDimX = %d\n",
            gridDimXMax, cHmlMaxGridDimX);
    exit(EXIT_FAILURE);
  }
  /* double gridDimX until gridDimY is no more than cHmlMaxGridDimY */
  for(gridDimX = gridDimXMin; gridDimX <= gridDimXMax; gridDimX *= 2) {
    gridDimY = (numBlocks + gridDimX - 1) / gridDimX;
    if(gridDimY <= cHmlMaxGridDimY) {
      break;
    }
  }
  if(gridDimX > gridDimXMax) {
    fprintf(stderr, "; Error: gridDimX > gridDimXMax\n");
    exit(EXIT_FAILURE);
  }
  grid->x = gridDimX;
  grid->y = gridDimY;
  grid->z = 1;         /* always = 1 */
}

void
hmlPagerankKernelArgSet(HmlPagerankKernelArg *kernelArgs,
                        uint32_t     numPartitions,
                        uint32_t    *minOutDeg, /* min out-degree of each partition */
                        uint32_t    *partitionPrefixSize) {
  uint32_t          p;
  uint32_t          numVertices;
  uint32_t          numBlocks;
  uint32_t          numThreadsPerBlock;

  for(p = 0; p < numPartitions; ++p) {
    kernelArgs[p].minDeg = minOutDeg[p];
    if(p < numPartitions - 1) {
      kernelArgs[p].maxDeg = minOutDeg[p + 1];
    }
    if(p > 0) {
      kernelArgs[p].minVertexRank = partitionPrefixSize[p - 1];
    }
    else {
      kernelArgs[0].minVertexRank = 0;
    }
    kernelArgs[p].maxVertexRank = partitionPrefixSize[p];
  }
  /* the last kernel instance always has maxDeg = infinity */
  kernelArgs[numPartitions - 1].maxDeg = (uint32_t)-1;

  /* setup the grid and block args for each partition */
  for(p = 0; p < numPartitions; ++p) {
    numVertices =
      kernelArgs[p].maxVertexRank - kernelArgs[p].minVertexRank; /* no +1 */
    switch(kernelArgs[p].id) {
    case 0:
      kernelArgs[p].block.x = 8;
      kernelArgs[p].block.y = 8;
      kernelArgs[p].block.z = 1;
      numThreadsPerBlock = kernelArgs[p].block.x * kernelArgs[p].block.y;
      numBlocks = (numVertices + numThreadsPerBlock - 1) / numThreadsPerBlock;
      /* it's OK to have gridDim.x be 1 for this kernel */
      hmlPagerankGridDimCalc(&kernelArgs[p].grid, numBlocks, 1, cHmlMaxGridDimX);
      break;

    case 1:
      //kernelArgs[p].block.x = MIN(kernelArgs[p].minDeg, cHmlThreadsPerWarp);
      kernelArgs[p].block.x = cHmlThreadsPerWarp;
      kernelArgs[p].block.y = 1;
      kernelArgs[p].block.z = 1;
      numBlocks = numVertices;
      /* gridDimX must >= 8, or CUDA behaves strangely */
      hmlPagerankGridDimCalc(&kernelArgs[p].grid, numBlocks, 8, cHmlMaxGridDimX);
      break;

    case 2:
      //kernelArgs[p].block.x = MIN(kernelArgs[p].minDeg, cHmlPagerankThreadsPerBlock);
      kernelArgs[p].block.x = cHmlPagerankThreadsPerBlock;
      kernelArgs[p].block.y = 1;
      kernelArgs[p].block.z = 1;
      numBlocks = numVertices;
      /* gridDimX must >= 8, or CUDA behaves strangely */
      hmlPagerankGridDimCalc(&kernelArgs[p].grid, numBlocks, 8, cHmlMaxGridDimX);
      break;

    case 3:
      kernelArgs[p].block.x = cHmlThreadsPerBlock;
      kernelArgs[p].block.y = kernelArgs[p].block.z = 1;
      kernelArgs[p].grid.x = cHmlBlocksPerGrid;
      kernelArgs[p].grid.y = kernelArgs[p].grid.z = 1;
      break;

    default:
      fprintf(stderr, "; Error: Unknown kernel id: %d\n", kernelArgs[p].id);
      exit(EXIT_FAILURE);
    }
  }
}

void
hmlPagerankKernelSetup(HmlPagerankKernel    *evenIterKernel,
                       HmlPagerankKernel    *oddIterKernel,
                       uint32_t               *minOutDegArr,
                       uint32_t                maxNumKernels,
                       bool                  useTextureMem,
                       uint32_t               *numPartitions,
                       HmlPagerankKernelArg *kernelArgArr) {
  assert(maxNumKernels >= 3);

  *numPartitions = 3;
  evenIterKernel[0] = useTextureMem ?
                      hmlPagerankKernelEvenIter0<true> : hmlPagerankKernelEvenIter0<false>;
  evenIterKernel[1] = useTextureMem ?
                      hmlPagerankKernelEvenIter1<true> : hmlPagerankKernelEvenIter1<false>;
  evenIterKernel[2] = useTextureMem ?
                      hmlPagerankKernelEvenIter2<true> : hmlPagerankKernelEvenIter2<false>;

  oddIterKernel[0] = useTextureMem ?
                     hmlPagerankKernelOddIter0<true> : hmlPagerankKernelOddIter0<false>;
  oddIterKernel[1] = useTextureMem ?
                     hmlPagerankKernelOddIter1<true> : hmlPagerankKernelOddIter1<false>;
  oddIterKernel[2] = useTextureMem ?
                     hmlPagerankKernelOddIter2<true> : hmlPagerankKernelOddIter2<false>;

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
