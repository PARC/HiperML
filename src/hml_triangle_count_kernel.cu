/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include "hml_triangle_count_kernel.h"

/* constants limited by CUDA */
#define cHmlBlocksPerGrid      128
#define cHmlThreadsPerBlock    128
#define cHmlTriangleCountLinearSearchMaxEdges 4096

/* 1-thread-1-row kernel assumes there are as many threads
 * as there are source vertices, and thus it does
 * NOT loop. It returns after processing at most
 * one vertex.
 * 'countArr' contains triangle count for each source vertex
 */
__global__ void
hmlTriangleCountKernel0(uint32_t        *countArr,   /* output count */
                        const uint32_t  *R,
                        const uint32_t  *E,
                        const uint32_t   maxSrcVertex,
                        const uint32_t  *vertexRank,
                        const uint32_t   minVertexRank, /* inclusive */
                        const uint32_t   maxVertexRank) /* exclusive */
{
  const int     x    = threadIdx.x + blockIdx.x * blockDim.x;
  const int     y    = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t  rank = x + y * blockDim.x * gridDim.x + minVertexRank;
  const uint32_t *eU;
  const uint32_t *eU2;
  const uint32_t *eV;
  const uint32_t *endU;
  const uint32_t *endV;
  uint32_t  u;
  uint32_t  v;
  uint32_t  w;
  uint32_t  numTriangles = 0;

  if (rank < maxVertexRank) {
    u = vertexRank[rank];
    endU = &E[R[u + 1]] - 1;
    for(eU = &E[R[u]]; eU < endU; eU++) {
      v = *eU;
      /* due to lexicographic edge pruning, v may be > maxSrcVertex */
      if(v <= maxSrcVertex) {
        eV = &E[R[v]];
        endV = &E[R[v + 1]] - 1;
        for(eU2 = eU + 1; eU2 <= endU; eU2++) {
          w = *eU2;
          while(eV <= endV && *eV < w) {
            eV++;
          }
          if(eV > endV) {
            break;
          }
          if(*eV == w) {
            numTriangles++;
          }
        }
      }
    }
    countArr[u] = numTriangles;
  }
}

/* 1-warp-1-row kernel assumes there are as many blocks
 * as there are source vertices, and each block is just
 * 32 threads. It does NOT loop, and returns after
 * processing at most one vertex.
 * 'countArr' contains triangle count for each source vertex
 */
__global__ void
hmlTriangleCountKernel1(uint32_t        *countArr,   /* output count */
                        const uint32_t  *R,
                        const uint32_t  *E,
                        const uint32_t   maxSrcVertex,
                        const uint32_t  *vertexRank,
                        const uint32_t   minVertexRank, /* inclusive */
                        const uint32_t   maxVertexRank) /* exclusive */
{
  const uint32_t  rank = blockIdx.x + blockIdx.y * gridDim.x + minVertexRank;
  const uint32_t  tid  = threadIdx.x;
  const uint32_t *eU;
  const uint32_t *eU2;
  const uint32_t *eV;
  const uint32_t *endU;
  const uint32_t *endV;
  uint32_t  u;
  uint32_t  v;
  uint32_t  w;
  uint32_t  numTriangles = 0;
  __shared__ uint32_t numTrianglesArr[cHmlThreadsPerWarp];

  if (rank < maxVertexRank) {
    u = vertexRank[rank];
    endU = &E[R[u + 1]] - 1;
    for(eU = &E[R[u]] + tid; eU < endU; eU += cHmlThreadsPerWarp) {
      v = *eU;
      /* due to lexicographic edge pruning, v may be > maxSrcVertex */
      if(v <= maxSrcVertex) {
        eV = &E[R[v]];
        endV = &E[R[v + 1]] - 1;
        for(eU2 = eU + 1; eU2 <= endU; eU2++) {
          w = *eU2;
          while(eV <= endV && *eV < w) {
            eV++;
          }
          if(eV > endV) {
            break;
          }
          if(*eV == w) {
            numTriangles++;
          }
        }
      }
    }
  }
  numTrianglesArr[tid] = numTriangles;
  //__syncthreads();
  for (int numReducers = blockDim.x / 2; numReducers > 0; numReducers /= 2) {
    if (tid < numReducers)
      numTrianglesArr[tid] += numTrianglesArr[tid + numReducers];
    //__syncthreads();
  }
  if (tid == 0 && rank < maxVertexRank)
      countArr[u] = numTrianglesArr[0];
}


/* 1-block-1-row kernel assumes there are as many blocks
 * as there are source vertices.
 * It does NOT loop, and returns after processing at most one vertex.
 * 'countArr' contains triangle count for each source vertex
 */
__global__ void
hmlTriangleCountKernel2(uint32_t        *countArr,   /* output count */
                        const uint32_t  *R,
                        const uint32_t  *E,
                        const uint32_t   maxSrcVertex,
                        const uint32_t  *vertexRank,
                        const uint32_t   minVertexRank, /* inclusive */
                        const uint32_t   maxVertexRank) /* exclusive */
{
  const uint32_t   rank = blockIdx.x + blockIdx.y * gridDim.x + minVertexRank;
  const uint32_t   tid  = threadIdx.x;
  const uint32_t *eU;
  const uint32_t *eU2;
  const uint32_t *eV;
  const uint32_t *endU;
  const uint32_t *endV;
  uint32_t  u;
  uint32_t  v;
  uint32_t  w;
  int32_t   lowV;
  int32_t   midV;
  int32_t   highV;
  uint32_t  numTriangles = 0;
  __shared__ uint32_t numTrianglesArr[cHmlTriangleCountThreadsPerBlock];

  if (rank < maxVertexRank) {
    u = vertexRank[rank];
    endU = &E[R[u + 1]] - 1;
    for(eU = &E[R[u]] + tid; eU < endU; eU += cHmlTriangleCountThreadsPerBlock) {
      v = *eU;
      /* due to lexicographic edge pruning, v may be > maxSrcVertex */
      if(v <= maxSrcVertex) {
        eV = &E[R[v]];
        endV = &E[R[v + 1]] - 1;
        if(endV - eV <= cHmlTriangleCountLinearSearchMaxEdges) {
          for(eU2 = eU + 1; eU2 <= endU; eU2++) {
            w = *eU2;
            while(eV <= endV && *eV < w) {
              eV++;
            }
            if(eV > endV) {
              break;
            }
            if(*eV == w) {
              numTriangles++;
            }
          }
        }
        else {  /* use binary search */
          for(eU2 = eU + 1; eU2 <= endU; eU2++) {
            w = *eU2;
            highV = (uint32_t)(endV - eV);
            if(highV <= cHmlTriangleCountLinearSearchMaxEdges) {
              while(eV <= endV && *eV < w) {
                eV++;
              }
              if(eV > endV) {
                break;
              }
              if(*eV == w) {
                numTriangles++;
              }
            }
            else {
              lowV = 0;
              while(lowV <= highV) {
                /* to avoid overflow in (lowV + highV) / 2 */
                midV = lowV + (highV - lowV) / 2;
                if(eV[midV] == w) {
                  lowV = midV + 1;
                  numTriangles++;
                  break;
                }
                else if(eV[midV] < w) {
                  lowV = midV + 1;
                }
                else {
                  highV = midV - 1;
                }
              }
              eV += lowV;
              if(eV > endV) {
                break;
              }
            }
          }
        }
      }
    }
  }
  numTrianglesArr[tid] = numTriangles;
  __syncthreads();
  for (int numReducers = blockDim.x / 2; numReducers > 0; numReducers /= 2) {
    if (tid < numReducers)
      numTrianglesArr[tid] += numTrianglesArr[tid + numReducers];
    __syncthreads();
  }
  if (tid == 0 && rank < maxVertexRank)
    countArr[u] = numTrianglesArr[0];
}

/* blockDim.x = cHmlTriangleCountSumThreadsPerBlock, .y = .z = 1
 * gridDim.x = cHmlTriangleCountSumBlocks, .y = .z = 1
 */
__global__ void
hmlTriangleCountSumKernel(uint64_t        *blockCountArr,   /* output count */
                          const uint32_t  *countArr,
                          const uint32_t   maxSrcVertex) {
  __shared__ uint64_t cache[cHmlTriangleCountSumThreadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIdx = threadIdx.x;
  uint64_t numTriangles = 0;

  while (tid <= maxSrcVertex) {
    numTriangles += countArr[tid];
    tid += blockDim.x * gridDim.x;
  }
  cache[cacheIdx] = numTriangles;
  __syncthreads();
  for (int numReducers = blockDim.x / 2; numReducers > 0; numReducers /= 2) {
    if (cacheIdx < numReducers)
      cache[cacheIdx] += cache[cacheIdx + numReducers];
    __syncthreads();
  }
  if (cacheIdx == 0)
    blockCountArr[blockIdx.x] = cache[0];
}

void
hmlTriangleCountKernelArgPrint(HmlTriangleCountKernelArg *kernelArgs, uint32_t numPartitions)
{
  for (uint32_t i = 0; i < numPartitions; ++i) {
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
hmlTriangleCountGridDimCalc(dim3   *grid,
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
hmlTriangleCountKernelArgSet(HmlTriangleCountKernelArg *kernelArgs,
                             uint32_t                     numPartitions,
                             uint32_t                    *minOutDeg,
                             uint32_t                    *partitionPrefixSize)
{
  uint32_t          p;
  uint32_t          numVertices;
  uint32_t          numBlocks;
  uint32_t          numThreadsPerBlock;

  for (p = 0; p < numPartitions; ++p) {
    kernelArgs[p].minDeg = minOutDeg[p];
    if (p < numPartitions - 1)
      kernelArgs[p].maxDeg = minOutDeg[p + 1];
    if (p > 0)
      kernelArgs[p].minVertexRank = partitionPrefixSize[p - 1];
    else
      kernelArgs[0].minVertexRank = 0;
    kernelArgs[p].maxVertexRank = partitionPrefixSize[p];
  }
  /* the last kernel instance always has maxDeg = infinity */
  kernelArgs[numPartitions - 1].maxDeg = (uint32_t)-1;

  /* setup the grid and block args for each partition */
  for (p = 0; p < numPartitions; ++p) {
    numVertices =
      kernelArgs[p].maxVertexRank - kernelArgs[p].minVertexRank; /* no +1 */
    switch (kernelArgs[p].id) {
    case 0:
      kernelArgs[p].block.x = 8;
      kernelArgs[p].block.y = 8;
      kernelArgs[p].block.z = 1;
      numThreadsPerBlock = kernelArgs[p].block.x * kernelArgs[p].block.y;
      numBlocks = (numVertices + numThreadsPerBlock - 1) / numThreadsPerBlock;
      /* it's OK to have gridDim.x be 1 for this kernel */
      hmlTriangleCountGridDimCalc(&kernelArgs[p].grid, numBlocks, 1, cHmlMaxGridDimX);
      break;

    case 1:
      //kernelArgs[p].block.x = MIN(kernelArgs[p].minDeg, cHmlThreadsPerWarp);
      kernelArgs[p].block.x = cHmlThreadsPerWarp;
      kernelArgs[p].block.y = 1;
      kernelArgs[p].block.z = 1;
      numBlocks = numVertices;
      /* gridDimX must >= 8, or CUDA behaves strangely */
      hmlTriangleCountGridDimCalc(&kernelArgs[p].grid, numBlocks, 8, cHmlMaxGridDimX);
      break;

    case 2:
      //kernelArgs[p].block.x = MIN(kernelArgs[p].minDeg, cHmlTriangleCountThreadsPerBlock);
      kernelArgs[p].block.x = cHmlTriangleCountThreadsPerBlock;
      kernelArgs[p].block.y = 1;
      kernelArgs[p].block.z = 1;
      numBlocks = numVertices;
      /* gridDimX must >= 8, or CUDA behaves strangely */
      hmlTriangleCountGridDimCalc(&kernelArgs[p].grid, numBlocks, 8, cHmlMaxGridDimX);
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
hmlTriangleCountKernelSetup(HmlTriangleCountKernel    *kernel,
                            uint32_t                    *minOutDegArr,
                            uint32_t                     maxNumKernels,
                            uint32_t                    *numPartitions,
                            HmlTriangleCountKernelArg *kernelArgArr)
{
  assert(maxNumKernels >= 3);

  *numPartitions = 3;
  kernel[0] = hmlTriangleCountKernel0;
  kernel[1] = hmlTriangleCountKernel1;
  kernel[2] = hmlTriangleCountKernel2;

  /* set CUDA cache preference for kernels
   * cudaFuncCachePreferL1 makes code run ~6% faster
   */
  HANDLE_ERROR(cudaFuncSetCacheConfig(kernel[0], cudaFuncCachePreferL1));
  HANDLE_ERROR(cudaFuncSetCacheConfig(kernel[1], cudaFuncCachePreferL1));
  HANDLE_ERROR(cudaFuncSetCacheConfig(kernel[2], cudaFuncCachePreferL1));

  /* setup minOutDeg array */
  minOutDegArr[0] = 1;
  minOutDegArr[1] = 32;
  minOutDegArr[2] = 1024;

  /* init kernel arg array */
  kernelArgArr[0].id = 0; /* use kernel id 0 for first partition */
  kernelArgArr[1].id = 1; /* use kernel id 1 for second partition */
  kernelArgArr[2].id = 2; /* use kernel id 2 for third partition */
}
