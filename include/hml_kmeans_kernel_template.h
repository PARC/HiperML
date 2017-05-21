/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_KMEANS_KERNEL_TEMPLATE_H_INCLUDED_
#define HML_KMEANS_KERNEL_TEMPLATE_H_INCLUDED_

#include <float.h>

extern texture<float, 1> texData1;

static __inline__ __device__ float tex1(const int32_t &i) {
  return tex1Dfetch(texData1, i);
}

/* The "for (k = 0; k < numClusts; ++k) loop" is NOT expanded */
template<uint32_t numDims, bool useTextureMem>
__global__ void hmlKmeansAssignCtrdSmem(uint32_t        *pAsmnts,
                               const float *pRows,
                               uint32_t         numRows,
                               const float *pCtrds,
                               uint32_t         numClusts)
{
  const uint32_t   tid = threadIdx.x;
  const uint32_t   rowStep = blockDim.x * blockDim.y * gridDim.x;
  const uint32_t   totalCtrdDims = numDims * numClusts;
  uint32_t   row;
  uint32_t   k;           /* cluster id */
  uint32_t   minK;        /* min-hmlKmeansDistance cluster id */
  uint32_t   dim;
  uint32_t   baseIdx;
  float  dist;
  float  minDist;
  float  delta;
  float  rowCache[numDims];
  const float*  pCtrdShared;
  extern __shared__ float  ctrdShared[];

  row = tid + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
  if (useTextureMem)
    baseIdx = numDims * row;
  else
    pRows += numDims * row;
  dim = tid;
  while (dim < totalCtrdDims) {
    ctrdShared[dim] = pCtrds[dim];
    dim += blockDim.x;
  }
  __syncthreads();
  while (row < numRows) {
    minDist = FLT_MAX;
    /* read row data and store in register memory */
#pragma unroll
    for (uint32_t dim = 0; dim < numDims; ++dim) {
      if (useTextureMem)
        rowCache[dim] = tex1(baseIdx + dim);
      else
        rowCache[dim] = pRows[dim];
    }
    pCtrdShared = ctrdShared;
    for (k = 0; k < numClusts; ++k) {
      dist = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta = rowCache[dim] - pCtrdShared[dim];
        dist += delta * delta;
      }
      pCtrdShared += numDims;
      if (dist < minDist) {
        minDist = dist;
        minK = k;
      }
    }
    pAsmnts[row] = minK;
    row += rowStep;
    if (useTextureMem)
      baseIdx += numDims * rowStep;
    else
      pRows += numDims * rowStep;
  }
}

/* The "for (k = 0; k < numClusts; ++k)" loop is expanded */
template<uint32_t numDims, bool useTextureMem>
__global__ void hmlKmeansAssignCtrdSmemUnroll(uint32_t        *pAsmnts,
                                     const float *pRows,
                                     uint32_t         numRows,
                                     const float *pCtrds,
                                     uint32_t         numClusts)
{
  const uint32_t   tid = threadIdx.x;
  const uint32_t   rowStep = blockDim.x * blockDim.y * gridDim.x;
  const uint32_t   totalCtrdDims = numDims * numClusts;
  /* round up numClusts to the nearest multiple of 16 */
  const uint32_t   numClusts16 = (numClusts + 15) / 16 * 16;
  const uint32_t   totalCtrdDims16 = numDims * numClusts16;
  uint32_t   row;
  uint32_t   k;           /* cluster id */
  uint32_t   minK;        /* min-hmlKmeansDistance cluster id */
  uint32_t   dim;
  uint32_t   baseIdx;
  float  dist[16];
  float  minDist;
  float  delta[16];
  float  rowCache[numDims];
  extern __shared__ float  ctrdShared[];

  row = tid + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
  if (useTextureMem)
    baseIdx = numDims * row;
  else
    pRows += numDims * row;
  /* load centroids into shared memory */
  dim = tid;
  while (dim < totalCtrdDims) {
    ctrdShared[dim] = pCtrds[dim];
    dim += blockDim.x;
  }
  /* pad the extra space with FLT_MAX / 2 */
  while (dim < totalCtrdDims16) {
    ctrdShared[dim] = FLT_MAX / 2;
    dim += blockDim.x;
  }
  __syncthreads();
  while (row < numRows) {
    minDist = FLT_MAX;
    /* read row data and store in register memory */
#pragma unroll
    for (uint32_t dim = 0; dim < numDims; ++dim) {
      if (useTextureMem)
        rowCache[dim] = tex1(baseIdx + dim);
      else
        rowCache[dim] = pRows[dim];
    }
    for (k = 0; k < numClusts; k += 16) {
      dist[0] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[0] = rowCache[dim] - ctrdShared[dim + k * numDims];
        dist[0] += delta[0] * delta[0];
      }
      if (dist[0] < minDist) {
        minDist = dist[0];
        minK = k;
      }
      dist[1] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[1] = rowCache[dim] - ctrdShared[dim + (k + 1) * numDims];
        dist[1] += delta[1] * delta[1];
      }
      if (dist[1] < minDist) {
        minDist = dist[1];
        minK = k + 1;
      }
      dist[2] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[2] = rowCache[dim] - ctrdShared[dim + (k + 2) * numDims];
        dist[2] += delta[2] * delta[2];
      }
      if (dist[2] < minDist) {
        minDist = dist[2];
        minK = k + 2;
      }
      dist[3] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[3] = rowCache[dim] - ctrdShared[dim + (k + 3) * numDims];
        dist[3] += delta[3] * delta[3];
      }
      if (dist[3] < minDist) {
        minDist = dist[3];
        minK = k + 3;
      }
      dist[4] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[4] = rowCache[dim] - ctrdShared[dim + (k + 4) * numDims];
        dist[4] += delta[4] * delta[4];
      }
      if (dist[4] < minDist) {
        minDist = dist[4];
        minK = k + 4;
      }
      dist[5] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[5] = rowCache[dim] - ctrdShared[dim + (k + 5) * numDims];
        dist[5] += delta[5] * delta[5];
      }
      if (dist[5] < minDist) {
        minDist = dist[5];
        minK = k + 5;
      }
      dist[6] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[6] = rowCache[dim] - ctrdShared[dim + (k + 6) * numDims];
        dist[6] += delta[6] * delta[6];
      }
      if (dist[6] < minDist) {
        minDist = dist[6];
        minK = k + 6;
      }
      dist[7] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[7] = rowCache[dim] - ctrdShared[dim + (k + 7) * numDims];
        dist[7] += delta[7] * delta[7];
      }
      if (dist[7] < minDist) {
        minDist = dist[7];
        minK = k + 7;
      }
      dist[8] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[8] = rowCache[dim] - ctrdShared[dim + (k + 8) * numDims];
        dist[8] += delta[8] * delta[8];
      }
      if (dist[8] < minDist) {
        minDist = dist[8];
        minK = k + 8;
      }
      dist[9] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[9] = rowCache[dim] - ctrdShared[dim + (k + 9) * numDims];
        dist[9] += delta[9] * delta[9];
      }
      if (dist[9] < minDist) {
        minDist = dist[9];
        minK = k + 9;
      }
      dist[10] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[10] = rowCache[dim] - ctrdShared[dim + (k + 10) * numDims];
        dist[10] += delta[10] * delta[10];
      }
      if (dist[10] < minDist) {
        minDist = dist[10];
        minK = k + 10;
      }
      dist[11] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[11] = rowCache[dim] - ctrdShared[dim + (k + 11) * numDims];
        dist[11] += delta[11] * delta[11];
      }
      if (dist[11] < minDist) {
        minDist = dist[11];
        minK = k + 11;
      }
      dist[12] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[12] = rowCache[dim] - ctrdShared[dim + (k + 12) * numDims];
        dist[12] += delta[12] * delta[12];
      }
      if (dist[12] < minDist) {
        minDist = dist[12];
        minK = k + 12;
      }
      dist[13] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[13] = rowCache[dim] - ctrdShared[dim + (k + 13) * numDims];
        dist[13] += delta[13] * delta[13];
      }
      if (dist[13] < minDist) {
        minDist = dist[13];
        minK = k + 13;
      }
      dist[14] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[14] = rowCache[dim] - ctrdShared[dim + (k + 14) * numDims];
        dist[14] += delta[14] * delta[14];
      }
      if (dist[14] < minDist) {
        minDist = dist[14];
        minK = k + 14;
      }
      dist[15] = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta[15] = rowCache[dim] - ctrdShared[dim + (k + 15) * numDims];
        dist[15] += delta[15] * delta[15];
      }
      if (dist[15] < minDist) {
        minDist = dist[15];
        minK = k + 15;
      }
    }
    pAsmnts[row] = minK;
    row += rowStep;
    if (useTextureMem)
      baseIdx += numDims * rowStep;
    else
      pRows += numDims * rowStep;
  }
}

/* The "for (k = 0; k < numClusts; ++k)" loop is NOT expanded
 * Note: Similar to hmlKmeansAssignCtrdSmemUnroll, one can implement
 * 'assignCtrdGmemExpk' that expands on the 'k < numClusts' loop,
 * but experiments show that it is always worse than
 * hmlKmeansAssignCtrdGmem
 */
template<uint32_t numDims, bool useTextureMem>
__global__ void hmlKmeansAssignCtrdGmem(uint32_t        *pAsmnts,
                               const float *pRows,
                               uint32_t         numRows,
                               const float *pCtrds,
                               uint32_t         numClusts)
{
  const int      x = threadIdx.x + blockIdx.x * blockDim.x;
  const int      y = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t   row = x + y * blockDim.x * gridDim.x;
  uint32_t   minK;        /* min-hmlKmeansDistance cluster id */
  uint32_t   baseIdx;
  float  dist;
  float  minDist;
  float  delta;
  float  rowCache[numDims];

  if (row < numRows)  {
    minDist = FLT_MAX;
    if (useTextureMem)
      baseIdx = numDims * row;
    else
      pRows += numDims * row;
#pragma unroll
    for (uint32_t dim = 0; dim < numDims; ++dim) {
      if (useTextureMem)
        rowCache[dim] = tex1(baseIdx + dim);
      else
        rowCache[dim] = pRows[dim];
    }
    for (uint32_t k = 0; k < numClusts; ++k) {
      dist = 0.0;
#pragma unroll
      for (uint32_t dim = 0; dim < numDims; ++dim) {
        delta = rowCache[dim] - pCtrds[dim];
        dist += delta * delta;
      }
      pCtrds += numDims;
      if (dist < minDist) {
        minDist = dist;
        minK = k;
      }
    }
    pAsmnts[row] = minK;
  }
}

/* invariant #1: gridDim.x == cHmlKmeansUpdateBlocks &&
 * invariant #2: blockDim.x == cHmlThreadsPerWarp
 * invariant #3: numDims = numDims0 + numDims1
 * Uses round robin strategy to sequentialize writes
 * that can be otherwise concurrent, if two threads
 * are processing points that are assigned to the same
 * cluster.
 * this kernel can use the entire warp to update a
 * single row, which is more efficient as numDims
 * gets bigger
 * numDims : # of dimensions of a row
 * numDims0: min(numDims, cHmlThreadsPerWarp) \in [0,32]
 * numDims1: max(0, numDims - cHmlThreadsPerWarp) \in [0, 32]
 */
template<uint32_t numDims, uint32_t numDims0, uint32_t numDims1, bool useTextureMem>
__global__ void
hmlKmeansUpdateCtrdGmem(float       *pCtrds,    /* numDims x numClusts x gridDim.x */
               uint32_t        *pSizes,    /* numClusts x gridDim.x */
               uint32_t         numClusts,
               const uint32_t  *pAsmnts,   /* numRows */
               const float *pRows,     /* numDims x numRows */
               uint32_t         numRows)
{
  const uint32_t    tid = threadIdx.x;
  const uint32_t    bid = blockIdx.x;
  const uint32_t    numBlocks = gridDim.x;
  const uint32_t    numRowsPerBlock = (numRows + numBlocks - 1) / numBlocks;
  const uint32_t    startRow = bid * numRowsPerBlock;
  int32_t     myNumRows;
  uint32_t    k;   /* cluster id */
  uint32_t    dim;
  uint32_t    baseIdx;
  float*  pCentroid;
  __shared__ uint32_t  kShared[cHmlThreadsPerWarp];

  /* sum along each dim for all points in the same cluster */
  if (useTextureMem)
    baseIdx = numDims * startRow;
  else
    pRows += numDims * startRow;
  pAsmnts += startRow;
  pCtrds += numDims * numClusts * bid;
  pSizes += numClusts * bid;
  myNumRows = (numRows > startRow) ? numRows - startRow : 0;
  myNumRows = min(numRowsPerBlock, myNumRows);
  int32_t T = myNumRows / cHmlThreadsPerWarp;
  for (int t = 0; t < T; ++t, pAsmnts += cHmlThreadsPerWarp) {
    kShared[tid] = pAsmnts[tid];
#pragma unroll
    for (int i = 0; i < cHmlThreadsPerWarp; ++i) {
      k = kShared[i];
      pCentroid = &pCtrds[numDims * k];
      if (tid == 0)
        ++pSizes[k];
      if (tid < numDims0) {
        if (useTextureMem)
          pCentroid[tid] += tex1(baseIdx + tid);
        else
          pCentroid[tid] += pRows[tid];
      }
      /* will be disabled by the compiler if numDims1 == 0 */
      if (numDims1 && tid < numDims1) {
        dim = tid + cHmlThreadsPerWarp;
        if (useTextureMem)
          pCentroid[dim] += tex1(baseIdx + dim);
        else
          pCentroid[dim] += pRows[dim];
      }
      if (useTextureMem)
        baseIdx += numDims;
      else
        pRows += numDims;
    }
  }
  //__syncthreads();
  /* process the last irregular tile, if any */
  int32_t tidEnd = myNumRows % cHmlThreadsPerWarp;
  if (tid < tidEnd)
    kShared[tid] = pAsmnts[tid];
  for (int i = 0; i < tidEnd; ++i) {
    k = kShared[i];
    pCentroid = &pCtrds[numDims * k];
    if (tid == 0)
      ++pSizes[k];
    if (tid < numDims0) {
      if (useTextureMem)
        pCentroid[tid] += tex1(baseIdx + tid);
      else
        pCentroid[tid] += pRows[tid];
    }
    /* will be disabled by the compiler if numDims1 == 0 */
    if (numDims1 && tid < numDims1) {
      dim = tid + cHmlThreadsPerWarp;
      if (useTextureMem)
        pCentroid[dim] += tex1(baseIdx + dim);
      else
        pCentroid[dim] += pRows[dim];
    }
    if (useTextureMem)
      baseIdx += numDims;
    else
      pRows += numDims;
  }
}

/* invariant #1: gridDim.x == cHmlKmeansUpdateBlocks &&
 * invariant #2: blockDim.x == cHmlThreadsPerWarp
 * invariant #3: numDims = numDims0 + numDims1
 * Uses round robin strategy to sequentialize writes
 * that can be otherwise concurrent, if two threads
 * are processing points that are assigned to the same
 * cluster.
 * this kernel can use the entire warp to update a
 * single row, which is more efficient as numDims
 * gets bigger
 * numDims : # of dimensions of a row
 * numDims0: min(numDims, cHmlThreadsPerWarp) \in [0,32]
 * numDims1: max(0, numDims - cHmlThreadsPerWarp) \in [0, 32]
 */
template<uint32_t numDims, uint32_t numDims0, uint32_t numDims1, bool useTextureMem>
__global__ void
hmlKmeansUpdateCtrdSmem(float       *pCtrds,    /* numDims x numClusts x gridDim.x */
               uint32_t        *pSizes,    /* numClusts x gridDim.x */
               uint32_t         numClusts,
               const uint32_t  *pAsmnts,   /* numRows */
               const float *pRows,     /* numDims x numRows */
               uint32_t         numRows)
{
  const uint32_t    tid = threadIdx.x;
  const uint32_t    bid = blockIdx.x;
  const uint32_t    numBlocks = gridDim.x;
  const uint32_t    numRowsPerBlock = (numRows + numBlocks - 1) / numBlocks;
  const uint32_t    totalCtrdDims = numDims * numClusts;
  const uint32_t    startRow = bid * numRowsPerBlock;
  int32_t     myNumRows;
  uint32_t    k;   /* cluster id */
  uint32_t    dim;
  uint32_t    baseIdx;
  float*  pCentroid;
  __shared__ uint32_t kShared[cHmlThreadsPerWarp];
  extern  __shared__ uint32_t shared[];
  uint32_t*  sizeShared = &shared[0];
  float*  ctrdShared = (float *)&shared[numClusts];

  /* sum along each dim for all points in the same cluster */
  if (useTextureMem)
    baseIdx = numDims * startRow;
  else
    pRows += numDims * startRow;
  pAsmnts += startRow;
  pCtrds += numDims * numClusts * bid;
  pSizes += numClusts * bid;
  /* init sizeShared */
  dim = tid;
  while (dim < numClusts) {
    sizeShared[dim] = 0;
    dim += blockDim.x; //must == cHmlThreadsPerWarp
  }
  /* init ctrdShared */
  dim = tid;
  while (dim < totalCtrdDims) {
    ctrdShared[dim] = 0.0;
    dim += blockDim.x; //must == cHmlThreadsPerWarp
  }
  myNumRows = (numRows > startRow) ? numRows - startRow : 0;
  myNumRows = min(numRowsPerBlock, myNumRows);
  int32_t T = myNumRows / cHmlThreadsPerWarp;
  //__syncthreads();
  for (int t = 0; t < T; ++t, pAsmnts += cHmlThreadsPerWarp) {
    kShared[tid] = pAsmnts[tid];
#pragma unroll
    for (int i = 0; i < cHmlThreadsPerWarp; ++i) {
      k = kShared[i];
      pCentroid = &ctrdShared[numDims * k];
      if (tid == 0)
        ++sizeShared[k];
      if (tid < numDims0) {
        if (useTextureMem)
          pCentroid[tid] += tex1(baseIdx + tid);
        else
          pCentroid[tid] += pRows[tid];
      }
      /* will be disabled by the compiler if numDims1 == 0 */
      if (numDims1 && tid < numDims1) {
        dim = tid + cHmlThreadsPerWarp;
        if (useTextureMem)
          pCentroid[dim] += tex1(baseIdx + dim);
        else
          pCentroid[dim] += pRows[dim];
      }
      if (useTextureMem)
        baseIdx += numDims;
      else
        pRows += numDims;
    }
  }
  //__syncthreads();
  /* process the last irregular tile, if any */
  int32_t tidEnd = myNumRows % cHmlThreadsPerWarp;
  if (tid < tidEnd)
    kShared[tid] = pAsmnts[tid];
  for (int i = 0; i < tidEnd; ++i) {
    k = kShared[i];
    pCentroid = &ctrdShared[numDims * k];
    if (tid == 0)
      ++sizeShared[k];
    if (tid < numDims0) {
      if (useTextureMem)
        pCentroid[tid] += tex1(baseIdx + tid);
      else
        pCentroid[tid] += pRows[tid];
    }
    /* will be disabled by the compiler if numDims1 == 0 */
    if (numDims1 && tid < numDims1) {
      dim = tid + cHmlThreadsPerWarp;
      if (useTextureMem)
        pCentroid[dim] += tex1(baseIdx + dim);
      else
        pCentroid[dim] += pRows[dim];
    }
    if (useTextureMem)
      baseIdx += numDims;
    else
      pRows += numDims;
  }
  //__syncthreads();
  /* write out pSizes and pCtrds */
  if (myNumRows > 0) {
    dim = tid;
    while (dim < numClusts) {
      pSizes[dim] = sizeShared[dim];
      dim += blockDim.x; //must == cHmlThreadsPerWarp
    }
    /* write out pCtrds */
    dim = tid;
    while (dim < totalCtrdDims) {
      pCtrds[dim] = ctrdShared[dim];
      dim += blockDim.x; //must == cHmlThreadsPerWarp
    }
  }
}

#endif /* HML_KMEANS_KERNEL_TEMPLATE_H_INCLUDED_ */
