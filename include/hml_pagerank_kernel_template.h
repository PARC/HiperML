#ifndef HML_PAGERANK_KERNEL_TEMPLATE_H_INCLUDED_
#define HML_PAGERANK_KERNEL_TEMPLATE_H_INCLUDED_
#include "hml_types.h"

extern texture<uint32_t, 1> texDataR;

extern texture<uint32_t, 1> texDataE;

extern texture<uint32_t, 1> texDataD;

extern texture<float, 1> texDataVec0;

extern texture<float, 1> texDataVec1;

static __inline__ __device__ uint32_t texR(const int32_t &i) {
  return tex1Dfetch(texDataR, i);
}

static __inline__ __device__ uint32_t texE(const int32_t &i) {
  return tex1Dfetch(texDataE, i);
}

static __inline__ __device__ uint32_t texD(const int32_t &i) {
  return tex1Dfetch(texDataD, i);
}

static __inline__ __device__ float texVec0(const int32_t &i) {
  return tex1Dfetch(texDataVec0, i);
}

static __inline__ __device__ float texVec1(const int32_t &i) {
  return tex1Dfetch(texDataVec1, i);
}

/* 1-thread-1-row kernel assumes there are as many threads
 * as there are source vertices, and thus it does
 * NOT loop. It returns after processing at most
 * one vertex.
 * map0 is the input map (i.e., current depth),
 *       which is accessed through texture
 * map1 is the output map (i.e, next depth)
 */
template<bool useTextureMem>
__global__ void
hmlPagerankKernelEvenIter0(float       *map1,         /* output map */
                           const uint32_t  *R,
                           const uint32_t  *E,
                           const uint32_t  *vertexRank,
                           const uint32_t   minVertexRank, /* inclusive */
                           const uint32_t   maxVertexRank, /* exclusive */
                           const float  dampingFactor) {
  const int     x    = threadIdx.x + blockIdx.x * blockDim.x;
  const int     y    = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t  rank = x + y * blockDim.x * gridDim.x + minVertexRank;

  if(rank < maxVertexRank) {
    float inputProbSum = 0.0;
    const uint32_t v = vertexRank[rank];
    if(useTextureMem) {
      uint32_t minSuccIdx = texR(v);
      const uint32_t maxSuccIdx = texR(v + 1);
      while(minSuccIdx < maxSuccIdx) {
        inputProbSum += texVec0(texE(minSuccIdx)) / texD(texE(minSuccIdx));
        minSuccIdx++;
      }
    }
    else {
      const uint32_t  *succ    = &E[R[v]];
      const uint32_t  *succMax = &E[R[v + 1]];
      while(succ < succMax) {
        inputProbSum += texVec0(*succ) / texD(*succ);
        succ++;
      }
    }
    map1[v] += dampingFactor * inputProbSum;
  }
}

/* 1-thread-1-row kernel assumes there are as many threads
 * as there are source vertices, and thus it does
 * NOT loop. It returns after processing at most
 * one vertex.
 * map1 is the input map (i.e., current depth)
 *       which is accessed through texture
 * map0 is the output map (i.e, next depth)
 */
template<bool useTextureMem>
__global__ void
hmlPagerankKernelOddIter0(float       *map0,         /* output map */
                          const uint32_t  *R,
                          const uint32_t  *E,
                          const uint32_t  *vertexRank,
                          const uint32_t   minVertexRank, /* inclusive */
                          const uint32_t   maxVertexRank, /* exclusive */
                          const float  dampingFactor) {
  const int     x    = threadIdx.x + blockIdx.x * blockDim.x;
  const int     y    = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t  rank = x + y * blockDim.x * gridDim.x + minVertexRank;

  if(rank < maxVertexRank) {
    float inputProbSum = 0.0;
    const uint32_t v = vertexRank[rank];
    if(useTextureMem) {
      uint32_t       minSuccIdx = texR(v);
      const uint32_t maxSuccIdx = texR(v + 1);
      while(minSuccIdx < maxSuccIdx) {
        inputProbSum += texVec1(texE(minSuccIdx)) / texD(texE(minSuccIdx));
        minSuccIdx++;
      }
    }
    else {
      const uint32_t  *succ    = &E[R[v]];
      const uint32_t  *succMax = &E[R[v + 1]];
      while(succ < succMax) {
        inputProbSum += texVec1(*succ) / texD(*succ);
        succ++;
      }
    }
    map0[v] += dampingFactor * inputProbSum;
  }
}

/* 1-warp-1-row kernel assumes there are as many blocks
 * as there are source vertices, and each block is just
 * 32 threads. It does NOT loop, and returns after
 * processing at most one vertex.
 * map0 is the input map (i.e., current depth)
 *       which is accessed through texture
 * map1 is the output map (i.e, next depth)
 */
template<bool useTextureMem>
__global__ void
hmlPagerankKernelEvenIter1(float       *map1,         /* output map */
                           const uint32_t  *R,
                           const uint32_t  *E,
                           const uint32_t  *vertexRank,
                           const uint32_t   minVertexRank, /* inclusive */
                           const uint32_t   maxVertexRank, /* exclusive */
                           const float  dampingFactor) {
  const uint32_t  rank = blockIdx.x + blockIdx.y * gridDim.x + minVertexRank;
  const uint32_t  tid  = threadIdx.x;
  uint32_t   v;
  float  psum = 0.0;
  __shared__ float  inputProbSum[cHmlThreadsPerWarp];

  if(rank < maxVertexRank) {
    v = vertexRank[rank];
    if(useTextureMem) {
      uint32_t       minSuccIdx = texR(v);
      const uint32_t maxSuccIdx = texR(v + 1);
      for(minSuccIdx += tid;
          minSuccIdx < maxSuccIdx;
          minSuccIdx += cHmlThreadsPerWarp) {
        psum += texVec0(texE(minSuccIdx)) / texD(texE(minSuccIdx));
      }
    }
    else {
      const uint32_t minSuccIdx = R[v];
      const uint32_t maxSuccIdx = R[v + 1];
      const uint32_t succs      = maxSuccIdx - minSuccIdx;
      const uint32_t succIdx    = minSuccIdx + tid;
      for(int succ = 0; tid + succ < succs; succ += cHmlThreadsPerWarp) {
        psum += texVec0(E[succIdx + succ]) / texD(E[succIdx + succ]);
      }
    }
  }
  inputProbSum[tid] = psum;
  //__syncthreads();
  for(int numReducers = blockDim.x / 2; numReducers > 0; numReducers /= 2) {
    if(tid < numReducers) {
      inputProbSum[tid] += inputProbSum[tid + numReducers];
    }
    //__syncthreads();
  }
  if(tid == 0 && rank < maxVertexRank) {
    map1[v] += dampingFactor * inputProbSum[0];
  }
}

/* 1-warp-1-row kernel assumes there are as many blocks
 * as there are source vertices, and each block is just
 * 32 threads. It does NOT loop, and returns after
 * processing at most one vertex.
 * map1 is the input map (i.e., current depth)
 *       which is accessed through texture
 * map0 is the output map (i.e, next depth)
 */
template<bool useTextureMem>
__global__ void
hmlPagerankKernelOddIter1(float       *map0,         /* output map */
                          const uint32_t  *R,
                          const uint32_t  *E,
                          const uint32_t  *vertexRank,
                          const uint32_t   minVertexRank, /* inclusive */
                          const uint32_t   maxVertexRank, /* exclusive */
                          const float  dampingFactor) {
  const uint32_t  rank = blockIdx.x + blockIdx.y * gridDim.x + minVertexRank;
  const uint32_t  tid  = threadIdx.x;
  uint32_t   v;
  float  psum = 0.0;
  __shared__ float  inputProbSum[cHmlThreadsPerWarp];

  if(rank < maxVertexRank) {
    v = vertexRank[rank];
    if(useTextureMem) {
      uint32_t       minSuccIdx = texR(v);
      const uint32_t maxSuccIdx = texR(v + 1);
      for(minSuccIdx += tid;
          minSuccIdx < maxSuccIdx;
          minSuccIdx += cHmlThreadsPerWarp) {
        psum += texVec1(texE(minSuccIdx)) / texD(texE(minSuccIdx));
      }
    }
    else {
      const uint32_t minSuccIdx = R[v];
      const uint32_t maxSuccIdx = R[v + 1];
      const uint32_t succs      = maxSuccIdx - minSuccIdx;
      const uint32_t succIdx    = minSuccIdx + tid;
      for(int succ = 0; tid + succ < succs; succ += cHmlThreadsPerWarp) {
        psum += texVec1(E[succIdx + succ]) / texD(E[succIdx + succ]);
      }
    }
  }
  inputProbSum[tid] = psum;
  //__syncthreads();
  for(int numReducers = blockDim.x / 2; numReducers > 0; numReducers /= 2) {
    if(tid < numReducers) {
      inputProbSum[tid] += inputProbSum[tid + numReducers];
    }
    //__syncthreads();
  }
  if(tid == 0 && rank < maxVertexRank) {
    map0[v] += dampingFactor * inputProbSum[0];
  }
}

/* 1-block-1-row kernel assumes there are as many blocks
 * as there are source vertices.
 * It does NOT loop, and returns after processing at most one vertex.
 * map0 is the input map (i.e., current depth)
 *       which is accessed through texture
 * map1 is the output map (i.e, next depth)
 */
template<bool useTextureMem>
__global__ void
hmlPagerankKernelEvenIter2(float       *map1,         /* output map */
                           const uint32_t  *R,
                           const uint32_t  *E,
                           const uint32_t  *vertexRank,
                           const uint32_t   minVertexRank, /* inclusive */
                           const uint32_t   maxVertexRank, /* exclusive */
                           const float  dampingFactor) {
  const uint32_t   rank = blockIdx.x + blockIdx.y * gridDim.x + minVertexRank;
  const uint32_t   tid  = threadIdx.x;
  uint32_t   v;
  float  psum = 0.0;
  __shared__ float  inputProbSum[cHmlPagerankThreadsPerBlock];

  if(rank < maxVertexRank) {
    v = vertexRank[rank];
    if(useTextureMem) {
      uint32_t       minSuccIdx = texR(v);
      const uint32_t maxSuccIdx = texR(v + 1);
      for(minSuccIdx += tid;
          minSuccIdx < maxSuccIdx;
          minSuccIdx += cHmlPagerankThreadsPerBlock) {
        psum += texVec0(texE(minSuccIdx)) / texD(texE(minSuccIdx));
      }
    }
    else {
      const uint32_t minSuccIdx = R[v];
      const uint32_t maxSuccIdx = R[v + 1];
      const uint32_t succs      = maxSuccIdx - minSuccIdx;
      const uint32_t succIdx    = minSuccIdx + tid;
      for(int succ = 0; tid + succ < succs; succ += cHmlPagerankThreadsPerBlock) {
        psum += texVec0(E[succIdx + succ]) / texD(E[succIdx + succ]);
      }
    }
  }
  inputProbSum[tid] = psum;
  __syncthreads();
  for(int numReducers = blockDim.x / 2; numReducers > 0; numReducers /= 2) {
    if(tid < numReducers) {
      inputProbSum[tid] += inputProbSum[tid + numReducers];
    }
    __syncthreads();
  }
  if(tid == 0 && rank < maxVertexRank) {
    map1[v] += dampingFactor * inputProbSum[0];
  }
}

/* 1-block-1-row kernel assumes there are as many blocks
 * as there are source vertices.
 * It does NOT loop, and returns after processing at most one vertex.
 * map1 is the input map (i.e., current depth)
 *       which is accessed through texture
 * map0 is the output map (i.e, next depth)
 */
template<bool useTextureMem>
__global__ void
hmlPagerankKernelOddIter2(float       *map0,         /* output map */
                          const uint32_t  *R,
                          const uint32_t  *E,
                          const uint32_t  *vertexRank,
                          const uint32_t   minVertexRank, /* inclusive */
                          const uint32_t   maxVertexRank, /* exclusive */
                          const float  dampingFactor) {
  const uint32_t   rank = blockIdx.x + blockIdx.y * gridDim.x + minVertexRank;
  const uint32_t   tid  = threadIdx.x;
  uint32_t   v;
  float  psum = 0.0;
  __shared__ float  inputProbSum[cHmlPagerankThreadsPerBlock];

  if(rank < maxVertexRank) {
    v = vertexRank[rank];
    if(useTextureMem) {
      uint32_t       minSuccIdx = texR(v);
      const uint32_t maxSuccIdx = texR(v + 1);
      for(minSuccIdx += tid;
          minSuccIdx < maxSuccIdx;
          minSuccIdx += cHmlPagerankThreadsPerBlock) {
        psum += texVec1(texE(minSuccIdx)) / texD(texE(minSuccIdx));
      }
    }
    else {
      const uint32_t minSuccIdx = R[v];
      const uint32_t maxSuccIdx = R[v + 1];
      const uint32_t succs      = maxSuccIdx - minSuccIdx;
      const uint32_t succIdx    = minSuccIdx + tid;
      for(int succ = 0; tid + succ < succs; succ += cHmlPagerankThreadsPerBlock) {
        psum += texVec1(E[succIdx + succ]) / texD(E[succIdx + succ]);
      }
    }
  }
  inputProbSum[tid] = psum;
  __syncthreads();
  for(int numReducers = blockDim.x / 2; numReducers > 0; numReducers /= 2) {
    if(tid < numReducers) {
      inputProbSum[tid] += inputProbSum[tid + numReducers];
    }
    __syncthreads();
  }
  if(tid == 0 && rank < maxVertexRank) {
    map0[v] += dampingFactor * inputProbSum[0];
  }
}

/* this kernel assumes there are cHmlBlocksPerGrid x
 * cHmlThreadsPerBlock threads, and if there are
 * more vertices than there are threads, then
 * each thread will process multiple vertices,
 * as implemented in the while() loop below
 * map0 is the input map (i.e., current depth)
 *       which is accessed through texture
 * map1 is the output map (i.e, next depth)
 */
template<bool useTextureMem>
__global__ void
hmlPagerankKernelEvenIter3(float       *map1,         /* output map */
                           const uint32_t  *R,
                           const uint32_t  *E,
                           const uint32_t  *vertexRank,
                           const uint32_t   minVertexRank, /* inclusive */
                           const uint32_t   maxVertexRank, /* exclusive */
                           const float  dampingFactor) {
  uint32_t   rank = threadIdx.x + blockIdx.x * blockDim.x + minVertexRank;

  while(rank < maxVertexRank) {
    float inputProbSum = 0.0;
    const uint32_t v = vertexRank[rank];
    if(useTextureMem) {
      uint32_t       minSuccIdx = texR(v);
      const uint32_t maxSuccIdx = texR(v + 1);
      while(minSuccIdx < maxSuccIdx) {
        inputProbSum += texVec0(texE(minSuccIdx)) / texD(texE(minSuccIdx));
        minSuccIdx++;
      }
    }
    else {
      const uint32_t *succ    = &E[R[v]];
      const uint32_t *succMax = &E[R[v + 1]];
      while(succ < succMax) {
        inputProbSum += texVec0(*succ) / texD(*succ);
        succ++;
      }
    }
    map1[v] += dampingFactor * inputProbSum;
    rank += blockDim.x * gridDim.x;
  }
}

/* this kernel assumes there are cHmlBlocksPerGrid x
 * cHmlThreadsPerBlock threads, and if there are
 * more vertices than there are threads, then
 * each thread will process multiple vertices,
 * as implemented in the while() loop below
 * map1 is the input map (i.e., current depth)
 *       which is accessed through texture
 * map0 is the output map (i.e, next depth)
 */
template<bool useTextureMem>
__global__ void
hmlPagerankKernelOddIter3(float       *map0,         /* output map */
                          const uint32_t  *R,
                          const uint32_t  *E,
                          const uint32_t  *vertexRank,
                          const uint32_t   minVertexRank, /* inclusive */
                          const uint32_t   maxVertexRank, /* exclusive */
                          const float  dampingFactor) {
  uint32_t   rank = threadIdx.x + blockIdx.x * blockDim.x + minVertexRank;

  while(rank < maxVertexRank) {
    float inputProbSum = 0.0;
    const uint32_t v = vertexRank[rank];
    if(useTextureMem) {
      uint32_t       minSuccIdx = texR(v);
      const uint32_t maxSuccIdx = texR(v + 1);
      while(minSuccIdx < maxSuccIdx) {
        inputProbSum += texVec1(texE(minSuccIdx)) / texD(texE(minSuccIdx));
        minSuccIdx++;
      }
    }
    else {
      const uint32_t *succ    = &E[R[v]];
      const uint32_t *succMax = &E[R[v + 1]];
      while(succ < succMax) {
        inputProbSum += texVec1(*succ) / texD(*succ);
        succ++;
      }
    }
    map0[v] += dampingFactor * inputProbSum;
    rank += blockDim.x * gridDim.x;
  }
}

/* dead code */
#if 0
/* this kernel assumes there are as many blocks
 * as there are source vertices, and thus it does
 * NOT loop. It returns after processing at most
 * one vertex.
 * there args for this kernel can be setup as follows:
 *   case 4:
 *     pHmlPagerankKernelArg[p].block.x = cHmlThreadsPerWarp;
 *     pHmlPagerankKernelArg[p].block.y = pHmlPagerankKernelArg[p].block.z = 1;
 *     numBlocks = (numVertices + cHmlThreadsPerWarp - 1) / cHmlThreadsPerWarp;
 *     // gridDimX must >= 8, or CUDA behaves strangely
 *     hmlPagerankGridDimCalc(&pHmlPagerankKernelArg[p].grid, numBlocks, 8, cHmlMaxGridDimX);
 *     break;
 */
__global__ void
pagerankEven4(uint32_t  *R,
              uint32_t  *E,
              uint32_t  *vertexRank,
              float *map0,
              float *map1,
              uint32_t   minVertexRank, /* inclusive */
              uint32_t   maxVertexRank, /* exclusive */
              float  dampingFactor) {
  uint32_t   tid = threadIdx.x;
  uint32_t   maxTid;
  uint32_t   rank = tid + blockIdx.x * blockDim.x +
                    blockIdx.y * blockDim.x * gridDim.x + minVertexRank;
  uint32_t   v;
  uint32_t  *succMin = NULL;
  uint32_t  *succMax = NULL;
  uint32_t  *succ = NULL;
  uint32_t   inputProbMinIdx;   /* inclusive */
  uint32_t   inputProbMaxIdx;   /* exclusive */
  uint32_t   curDeg;
  uint32_t   doneDeg = 0;
  uint32_t   offset = 1;        /* for scan */
  float  curProbSum;
  bool     needPrev; /* need to add 'prevProbSum' to curProbSum */
  bool     giveNext; /* need to set 'nextProbSum' to curProbSum */
  __shared__ uint32_t   totalDeg;
  __shared__ float  prevProbSum; /* from previous iteration */
  __shared__ float  nextProbSum; /* for next iteration */
  __shared__ uint32_t   prefixDeg[cHmlThreadsPerWarp];
  __shared__ uint32_t  *localR[cHmlThreadsPerWarp];
  __shared__ float  inputProb[cHmlThreadsPerWarp];

  prefixDeg[tid] = 0;
  if(rank < maxVertexRank) {
    v = vertexRank[rank];
    succMin = succ = &E[R[v]];
    succMax = &E[R[v + 1]];
    prefixDeg[tid] = (succMax - succMin) / 2; /* each page takes 2 words */
  }
  /* preform exclusive scan */
  /* build sum in place up the tree */
  for(uint32_t d = cHmlThreadsPerWarp >> 1; d > 0; d >>= 1) {
    //__syncthreads();
    if(tid < d) {
      uint32_t ai = offset*(2*tid+1)-1;
      uint32_t bi = offset*(2*tid+2)-1;
      prefixDeg[bi] += prefixDeg[ai];
    }
    offset *= 2;
  }
  /* clear the last element */
  if(tid == 0) {
    totalDeg = prefixDeg[cHmlThreadsPerWarp - 1];
    prefixDeg[cHmlThreadsPerWarp - 1] = 0;
  }
  /* traverse down tree & build scan */
  for(uint32_t d = 1; d < cHmlThreadsPerWarp; d *= 2) {
    offset >>= 1;
    //__syncthreads();
    if(tid < d) {
      uint32_t ai = offset*(2*tid+1)-1;
      uint32_t bi = offset*(2*tid+2)-1;
      uint32_t tmp = prefixDeg[ai];
      prefixDeg[ai] = prefixDeg[bi];
      prefixDeg[bi] += tmp;
    }
  }
  //__syncthreads();
  curDeg = prefixDeg[tid];
  while(doneDeg < totalDeg) {
    inputProbMinIdx = inputProbMaxIdx = (uint32_t)-1;
    /* is this the first successor for this iteration and at least
     * one of its siblings was processed in the previous iteration?
     * if so, needPrev = true and later on 'prevProbSum' will be
     * added to 'curProbSum'
     */
    needPrev = (curDeg == doneDeg && succ < succMax && succ != succMin) ?
               true : false;
    prevProbSum = (needPrev == true) ? nextProbSum : 0.0;
    /* distribute the successors to the entire warp */
    while((curDeg < doneDeg + cHmlThreadsPerWarp) && (succ < succMax)) {
      localR[curDeg - doneDeg] = succ;
      if(inputProbMinIdx == (uint32_t)-1) {
        inputProbMinIdx = curDeg - doneDeg;  /* dub as flag being in this loop */
      }
      succ += 2;
      ++curDeg;
    }
    /* set inputProbMaxIdx only if the while loop above is executed */
    if(inputProbMinIdx != (uint32_t)-1) {
      inputProbMaxIdx = curDeg - doneDeg;  /* ..MaxIdx is exclusive */
    }
    /* is this the last successor for this iteration and at least
     * one of its siblings will be processed in the next iteration?
     * if so, giveNext = true and later on 'nextProbSum' will be
     * set to 'curProbSum'
     */
    giveNext = ((curDeg - doneDeg == cHmlThreadsPerWarp) && (succ < succMax)) ?
               true : false;
    /* done with distributing successors among threads in the same warp */
    //__syncthreads();
    maxTid = MIN(totalDeg - doneDeg, cHmlThreadsPerWarp);
    if(tid < maxTid) {
      inputProb[tid] = texVec0(*localR[tid]] * (*(float *)(localR[tid] + 1));
    }
    doneDeg += cHmlThreadsPerWarp;
    //__syncthreads();
    /* PageRank (possibly partial) reduction */
    /* is there a 'inputProb' that this thread is responsible for adding? */
    if(inputProbMinIdx != (uint32_t)-1) {
      curProbSum = 0.0;
      while(inputProbMinIdx < inputProbMaxIdx) {
        curProbSum += inputProb[inputProbMinIdx];
        ++inputProbMinIdx;
      }
      curProbSum += prevProbSum;
      if(!giveNext) { /* no need to check validity of 'v' if we are here */
        map1[v] += dampingFactor * curProbSum;  /* reduction completed */
      }
      else { /* store result of partial reduction to a __shared__ variable */
        nextProbSum = curProbSum;
      }
    }
    //__syncthreads();
  }
}

uint32_t *gR = NULL;

bool
hmlGraphSortVertexIdxByOutDegComparator(const uint32_t idx1, const uint32_t idx2) {
  return (gR[idx1 + 1] - gR[idx1]) < (gR[idx2 + 1] - gR[idx2]);
}

/* vidArr is a pre-allocated array of size:
 * (graph->maxSrcVertex - graph->minSrcVertex + 1)
 * Upon return, vidArr stores the vertex ids ordered by
 * increasing out-degree
 */
void
hmlGraphSortVertexIdxByOutDeg(HmlGraph *graph, uint32_t *vidArr) {
  uint32_t   idx;
  uint32_t  *vidArrEnd;
  uint32_t   numSrcVertices = graph->maxSrcVertex - graph->minSrcVertex + 1;

  vidArrEnd = vidArr + numSrcVertices;
  /* init vidArr */
  for(idx = 0; idx < numSrcVertices; ++idx) {
    vidArr[idx] = idx + graph->minSrcVertex;
  }
  /* set global variable gR needed by the comparator*/
  gR = graph->R;
  /* perform index sort */
  sort(vidArr, vidArrEnd, hmlGraphSortVertexIdxByOutDegComparator);
}

#endif /* dead code */

#endif /* HML_PAGERANK_KERNEL_TEMPLATE_H_INCLUDED_ */
