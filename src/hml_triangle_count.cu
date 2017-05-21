/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include "hml_quick_sort.h"
#include "hml_triangle_count.h"
#include "hml_file_utils.h"

#define cHmlBlocksPerGrid      128
#define cHmlThreadsPerBlock    128

#define cHmlTriangleCountPartitionSizeInit      1024
#define cHmlTriangleCountMaxNumPartitions       32
#define cHmlTriangleCountLinearSearchMaxEdges   4096
#define cHmlTriangleCountNumBlocks              256

typedef struct {
  uint32_t *P; /* vertex permutation array */
} HmlTriangleCountGraphInsertState;

/* partition 'graph' into 'numPartitions' s.t. for all vertices
 * v in partition p, the following property holds:
 * minOutDeg[p] <= out-degree(v) < minOutDeg[p + 1],
 * except for the last partition p = numParititions - 1, for which
 * it holds:
 * minOutDeg[numPartitions - 1] <= out-degree(v) < infinity
 * Thus, minOutDeg[] has 'numPartitions' elements.
 * The output is stored in vertexRank, an array of size
 * (graph->maxSrcVertex - graph->minSrcVertex + 1) elements.
 * vertexRank[] must be allocated by the caller of this function.
 * vertexRank[r] stores the id of vertex v in partition p s.t.
 * vertexRank[r] == v && partitionPrefixSize[p - 1] <= r &&
 * r < partitionPrefixSize[p], except for the first partition p = 0, where
 * 0 <= r < partitionPrefixSize[0]
 * The actual size of vertexRank is given by:
 * partitionPrefixSize[numPartitions - 1], which should never exceed
 * numSrcVertices (see below). It's the caller's responsibility to
 * resize vertexRank afterwards to free its unused portion.
 */
void
hmlTriangleCountPartitionVertexByOutDeg(HmlGraphCore  *core,
                                        uint32_t *minOutDeg,
                                        uint32_t  numPartitions,
                                        uint32_t *vertexRank,
                                        uint32_t *partitionPrefixSize)
{
  uint32_t **partitions;
  uint32_t   p;
  uint32_t   v;
  uint32_t   outDeg;
  uint32_t  *R = core->R;
  uint32_t  *pPartitionAllocSize;   /* allocation size */
  uint32_t **partitionPtr;
  uint32_t **partitionEndPtr;        /* actual used size */
  uint32_t   numSrcVertices = core->maxSrcVertex - core->minSrcVertex + 1;
  uint32_t   prefixSize = 0;

  MALLOC(partitions, uint32_t *, numPartitions);
  MALLOC(partitionPtr, uint32_t *, numPartitions);
  MALLOC(partitionEndPtr, uint32_t *, numPartitions);
  MALLOC(pPartitionAllocSize, uint32_t, numPartitions);
  for (p = 0; p < numPartitions; ++p) {
    MALLOC(partitions[p], uint32_t, cHmlTriangleCountPartitionSizeInit);
    pPartitionAllocSize[p] = cHmlTriangleCountPartitionSizeInit;
    partitionPtr[p] = partitions[p];
    partitionEndPtr[p] = partitions[p] + cHmlTriangleCountPartitionSizeInit;
  }
  for (v = core->minSrcVertex; v <= core->maxSrcVertex; ++v) {
    outDeg = R[v + 1] - R[v]; /* each page takes one 32-bit word */
    /* use linear scan to find which partition this vertex belongs to */
    for (p = 0; p < numPartitions && minOutDeg[p] <= outDeg; ++p);
    if (p > 0) {
      --p;
      if (partitionPtr[p] == partitionEndPtr[p]) {
        REALLOC(partitions[p], uint32_t, pPartitionAllocSize[p] * 2);
        partitionPtr[p] = partitions[p] + pPartitionAllocSize[p];
        pPartitionAllocSize[p] *= 2;
        partitionEndPtr[p] = partitions[p] + pPartitionAllocSize[p];
      }
      *partitionPtr[p]++ = v;
    }
  }
  for (p = 0; p < numPartitions; ++p) {
    prefixSize += partitionPtr[p] - partitions[p];
    partitionPrefixSize[p] = prefixSize;
    if (prefixSize > numSrcVertices) {
      fprintf(stderr, "; Error: prefixSize = %d > numSrcVertices = %d\n",
              prefixSize, numSrcVertices);
      exit(EXIT_FAILURE);
    }
    memcpy((void*)vertexRank, partitions[p],
           sizeof(uint32_t) * (partitionPtr[p] - partitions[p]));
    vertexRank += partitionPtr[p] - partitions[p];
  }
  /* free memory */
  for (p = 0; p < numPartitions; ++p) {
    FREE(partitions[p]);
    FREE(partitionPtr);
    FREE(partitionEndPtr);
    FREE(pPartitionAllocSize);
  }
  FREE(partitions);
}

static HmlErrCode
hmlTriangleCountCopyToGpu(HmlTriangleCountBase *cpu, HmlTriangleCountBase *gpu)
{
  HML_ERR_PROLOGUE;

  /* shallow copy of HmlTriangleCount object from "cpu" to "gpu" */
  memcpy(gpu, cpu, sizeof(HmlTriangleCount));
  /* reset all pointer members of "gpu" object */
  gpu->D = NULL;
  gpu->P = NULL;
  gpu->numTrianglesEachThread = NULL;
  /* alloc memory on GPU for graph core and copy the graph data */
  hmlGraphCoreCopyToGpu(&cpu->core, &gpu->core);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTriangleCountGpuInit(HmlTriangleCount *triangleCount) {
  HML_ERR_PROLOGUE;
  HmlGraphCore *core = &triangleCount->cpu.core;
  uint32_t   minSrcVertex = core->minSrcVertex;
  uint32_t   maxSrcVertex = core->maxSrcVertex;
  uint32_t   numSrcVertices = maxSrcVertex - minSrcVertex + 1;
  uint32_t   numVertices = max(maxSrcVertex, core->maxDestVertex) + 1;
  uint32_t  *vertexRank; /* array of vertex ids sorted by out-degree */
  uint32_t   minOutDeg[cHmlTriangleCountMaxNumPartitions]; /* min out-deg of each partition */
  uint32_t   partitionPrefixSize[cHmlTriangleCountMaxNumPartitions]; /* cumulative size */
  uint32_t   vertexRankSize;
  size_t   freeBytes;
  size_t   totalBytes;
  double   cpuStart, wallStart;
  double   cpuEnd, wallEnd;

  /* get free gpu memory size */
  if (triangleCount->verbosity >= 2) {
    HANDLE_ERROR(cudaMemGetInfo(&freeBytes, &totalBytes));
    fprintf(stderr, "; Info: GPU memory: %ld bytes free, %ld bytes total\n",
            freeBytes, totalBytes);
  }
  hmlGetSecs(&cpuStart, &wallStart);
  /* create GPU object */
  hmlTriangleCountCopyToGpu(&triangleCount->cpu, &triangleCount->gpu);
  cudaDeviceSynchronize();
  hmlGetSecs(&cpuEnd, &wallEnd);
  if (triangleCount->verbosity >= 2) {
    fprintf(stderr, "; Info: Load graph to device: wall time = %.2lf\n",
            (wallEnd - wallStart) * 1000);
  }
  if (numVertices > cHmlMaxCudaTexture1DLinear) {
    hmlPrintf("; Error: Number of vertices exceeds the maximum "
              "texture 1D size\n");
    HML_ERR_GEN(true, cHmlErrGeneral);
  }
  hmlTriangleCountKernelSetup(triangleCount->kernel, minOutDeg,
                              cHmlTriangleCountMaxNumKernels,
                              &triangleCount->numPartitions,
                              triangleCount->kernelArgs);

  /* create vertexRank mapping */
  hmlGetSecs(&cpuStart, &wallStart);
  /* allocate vertexRank[] on CPU */
  MALLOC(vertexRank, uint32_t, numSrcVertices);
  hmlTriangleCountPartitionVertexByOutDeg(&triangleCount->cpu.core, minOutDeg,
                                     triangleCount->numPartitions,
                                     vertexRank, partitionPrefixSize);
  //hmlGetSecs(&cpuEnd, &wallEnd);
  //fprintf(stderr, "; Info: Partition vertices on CPU: "
  //      "cpu time = %.2lf, wall time = %.2lf\n",
  //      (cpuEnd - cpuStart* 1000, (wallEnd - wallStart) * 1000);

  vertexRankSize = partitionPrefixSize[triangleCount->numPartitions - 1];
  /* resize vertexRank */
  REALLOC(vertexRank, uint32_t, vertexRankSize);
  /* allocate gpuVertexRank[] on device */
  HANDLE_ERROR(cudaMalloc(&triangleCount->gpuVertexRank,
                          sizeof(uint32_t) * vertexRankSize));
  /* copy vertexRank[] to gpuVertexRank[] */
  HANDLE_ERROR(cudaMemcpy(triangleCount->gpuVertexRank,
                          vertexRank,
                          sizeof(uint32_t) * vertexRankSize,
                          cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  hmlGetSecs(&cpuEnd, &wallEnd);
  if (triangleCount->verbosity >= 2) {
    fprintf(stderr, "; Info: Partition and copy vertice ranks to device: "
            "wall time = %.2lf\n", (wallEnd - wallStart) * 1000);
    fprintf(stderr, "; Info: Number of pages with in-coming link: %d (%.2lf%%)\n",
            vertexRankSize, 100 * vertexRankSize/(double)(numVertices));
    fprintf(stderr, "; Info: Partitioned graph size = %.2lf MB\n",
            (core->maxNumSrcVertices + core->numEdges + vertexRankSize) *
            sizeof(uint32_t) / (double)(1024 * 1024));
  }
  /* print vertex ranks for small graphs */
  if (triangleCount->verbosity >= 3 && vertexRankSize <= 100) {
    for (uint32_t r = 0; r < vertexRankSize; ++r) {
      fprintf(stderr, "; Info: rank %3d = vertex %3d\n", r, vertexRank[r]);
    }
  }

  /* set the kernel arguments */
  hmlTriangleCountKernelArgSet(triangleCount->kernelArgs, triangleCount->numPartitions,
                          minOutDeg, partitionPrefixSize);

  /* print kernel params */
  if (triangleCount->verbosity >= 2)
    hmlTriangleCountKernelArgPrint(triangleCount->kernelArgs, triangleCount->numPartitions);

  HANDLE_ERROR(cudaMalloc(&triangleCount->gpuCountArr,
                          sizeof(uint32_t) * (maxSrcVertex + 1)));
  HANDLE_ERROR(cudaMemset(triangleCount->gpuCountArr, 0,
                          sizeof(uint32_t) * (maxSrcVertex + 1)));
  HANDLE_ERROR(cudaMalloc(&triangleCount->gpuBlockCountArr,
                          sizeof(uint64_t) * cHmlTriangleCountSumBlocks));
  HANDLE_ERROR(cudaMemset(triangleCount->gpuBlockCountArr, 0,
                          sizeof(uint64_t) * cHmlTriangleCountSumBlocks));
  FREE(vertexRank);
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTriangleCountGpu(HmlTriangleCount *triangleCount) {
  HML_ERR_PROLOGUE;
  HmlGraphCore *gpuCore = &triangleCount->gpu.core;
  uint32_t   maxSrcVertex = gpuCore->maxSrcVertex;
  uint32_t  *gpuVertexRank = triangleCount->gpuVertexRank;
  uint32_t  *gpuCountArr = triangleCount->gpuCountArr;
  uint64_t  *gpuBlockCountArr = triangleCount->gpuBlockCountArr;
  uint32_t   numPartitions = triangleCount->numPartitions;
  double   cpuStart, wallStart;
  double   cpuEnd, wallEnd;
  HmlTriangleCountKernel    *kernel = triangleCount->kernel;
  HmlTriangleCountKernelArg *kernelArgs = triangleCount->kernelArgs;
  int      blk;
  uint32_t  *countArr;

  MALLOC(countArr, uint32_t, maxSrcVertex + 1);
  hmlGetSecs(&cpuStart, &wallStart);
  //fprintf(stderr, "; Info: iter = %d\n", iter);
  for (uint32_t p = 0; p < numPartitions; ++p) {
    kernel[kernelArgs[p].id]<<<kernelArgs[p].grid, kernelArgs[p].block>>>
      (gpuCountArr, gpuCore->R, gpuCore->E, maxSrcVertex, gpuVertexRank,
       kernelArgs[p].minVertexRank, kernelArgs[p].maxVertexRank);
  }
  hmlTriangleCountSumKernel<<<cHmlTriangleCountSumBlocks,
      cHmlTriangleCountSumThreadsPerBlock>>>
      (gpuBlockCountArr, gpuCountArr, maxSrcVertex);
  cudaDeviceSynchronize();
  hmlGetSecs(&cpuEnd, &wallEnd);
  if (triangleCount->verbosity >= 1) {
    fprintf(stderr, "; Info: GPU TriangleCount: wall time = %.2lf\n",
            (wallEnd - wallStart) * 1000);
  }
  HANDLE_ERROR(cudaMemcpy(triangleCount->blockCountArr,
                          gpuBlockCountArr,
                          sizeof(uint64_t) * cHmlTriangleCountSumBlocks,
                          cudaMemcpyDeviceToHost));
  triangleCount->gpu.numTriangles = 0;
  for (blk = 0; blk < cHmlTriangleCountSumBlocks; blk++) {
    triangleCount->gpu.numTriangles += triangleCount->blockCountArr[blk];
  }
  /*
  HANDLE_ERROR(cudaMemcpy(countArr,
                          gpuCountArr,
                          sizeof(uint32_t) * (maxSrcVertex + 1),
                          cudaMemcpyDeviceToHost));
  triangleCount->gpu.numTriangles = 0;
  for (blk = 0; blk <= maxSrcVertex; blk++) {
    triangleCount->gpu.numTriangles += countArr[blk];
  }
  */
  FREE(countArr);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTriangleCountBaseInit(HmlTriangleCountBase *count, uint32_t numThreads) {
  HML_ERR_PROLOGUE;
  memset(count, 0, sizeof(HmlTriangleCount));
  HML_ERR_PASS(hmlVertexPartitionInit(&count->partition));
  count->numThreads = numThreads;
  if(numThreads > 1) {
    CALLOC(count->numTrianglesEachThread, uint64_t, numThreads);
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTriangleCountBaseDelete(HmlTriangleCountBase *count) {
  HML_ERR_PROLOGUE;
  HmlGraphCore *core = &count->core;

  if(count->numThreads > 1) {
    FREE(count->numTrianglesEachThread);
  }
  FREE(count->D);
  FREE(count->P);
  HML_ERR_PASS(hmlGraphCoreDelete(core));
  HML_ERR_PASS(hmlVertexPartitionDelete(&count->partition));
  memset(count, 0, sizeof(HmlTriangleCount));

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTriangleCountInit(HmlTriangleCount *triangleCount) {
  HML_ERR_PROLOGUE;
  memset(triangleCount, 0, sizeof(HmlTriangleCount));

  HML_ERR_PASS(hmlTriangleCountBaseInit(&triangleCount->cpu, 1));
  MALLOC(triangleCount->blockCountArr, uint64_t,
         cHmlTriangleCountSumBlocks);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTriangleCountDelete(HmlTriangleCount *triangleCount) {
  HML_ERR_PROLOGUE;

  FREE(triangleCount->blockCountArr);
  HML_ERR_PASS(hmlTriangleCountBaseDelete(&triangleCount->cpu));
  /* free GPU stuff */
  HANDLE_ERROR(cudaFree(triangleCount->gpuCountArr));
  HANDLE_ERROR(cudaFree(triangleCount->gpuBlockCountArr));
  if (triangleCount->gpu.core.numEdges > 0) {
    hmlGraphCoreGpuDelete(&triangleCount->gpu.core);
  }
  HANDLE_ERROR(cudaFree(triangleCount->gpuVertexRank));
  memset(triangleCount, 0, sizeof(HmlTriangleCount));

  HML_NORMAL_RETURN;
}

/* only allows edge (u, v), iff u < v */
static HmlErrCode
hmlTriangleCountGraphAppender(HmlGraphCore  *core,
                              void          *appendState,
                              uint32_t       srcVertex,
                              uint32_t       destVertex) {
  HML_ERR_PROLOGUE;
  HmlGraphCoreAppendState *appState = (HmlGraphCoreAppendState *)appendState;

  if(srcVertex < destVertex) {
      HML_ERR_PASS(hmlGraphCoreAppend(core, appState, srcVertex, destVertex));
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTriangleCountReadOrderedTsv2File(HmlTriangleCountBase *count,
                                    FILE *file,
                                    bool srcVertexOnRightColumn) {
  HML_ERR_PROLOGUE;
  HmlGraphCoreAppendState state;

  hmlGraphCoreAppendFromTsv2File(&count->core, file, srcVertexOnRightColumn,
                                 (HmlGraphCoreAppendIniter)hmlGraphCoreAppendInit,
                                 (HmlGraphCoreAppender)hmlTriangleCountGraphAppender,
                                 (HmlGraphCoreAppendFinalizer)hmlGraphCoreAppendFinal,
                                 &state);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTriangleCountReadOrderedTsv2FileByName(HmlTriangleCountBase *count,
                                          char const *fileName,
                                          bool srcVertexOnRightColumn) {
  HML_ERR_PROLOGUE;
  FILE *file;

  HML_ERR_PASS(hmlFileOpenRead(fileName, &file));
  HML_ERR_PASS(hmlTriangleCountReadOrderedTsv2File(count, file,
               srcVertexOnRightColumn));
  HML_ERR_PASS(hmlFileClose(file));

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlTriangleCountGraphInserter(HmlGraphCore  *core,
                              void          *insertState,
                              uint32_t         srcVertex,
                              uint32_t         destVertex) {
  HML_ERR_PROLOGUE;
  HmlTriangleCountGraphInsertState *state
    = (HmlTriangleCountGraphInsertState *)insertState;
  uint32_t tmpVertex;

  if (srcVertex != destVertex) {
    srcVertex = state->P[srcVertex];
    destVertex = state->P[destVertex];
    if (srcVertex > destVertex) {
      tmpVertex = srcVertex;
      srcVertex = destVertex;
      destVertex = tmpVertex;
    }
    HML_ERR_PASS(hmlGraphCoreAddEdge(core, srcVertex, destVertex));
  }

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlTriangleCountSetPartition(HmlTriangleCountBase *count) {
  HML_ERR_PROLOGUE;
  uint64_t maxNumEdgesPerPartition =
    (count->core.numEdges + count->numThreads - 1) / count->numThreads;

  maxNumEdgesPerPartition += count->core.maxOutDegree;
  HML_ERR_PASS(hmlGraphCoreVertexPartition(&count->core,
               maxNumEdgesPerPartition, &count->partition));

  HML_NORMAL_RETURN;
}

static int
HML_QSORT_COMPARE_FUNC(hmlTriangleCountVertexCompare, a, b, arg) {
  uint32_t *D = (uint32_t *)arg;
  return D[*(const uint32_t *)b] - D[*(const uint32_t *)a];
}

/* reorder the edges of an undirected graph such that
 * the vertices with more neighbors are stored
 * as source vertices; whereas those with fewer
 * neighbors are stored in the adjacency list of
 * the high-degree vertices.
 * that is, if (u, v) in E and deg(u) > deg(v), then
 * v is stored as a successor of u but NOT the other
 * way around in a succinct encoding of undirected graph
 */
HmlErrCode
hmlTriangleCountReorderEdges(HmlTriangleCountBase *count) {
  HML_ERR_PROLOGUE;
  HmlTriangleCountGraphInsertState insertState;
  HmlGraphCore *core = &count->core;
  HmlGraphCore  copyVal;
  HmlGraphCore *copy = &copyVal;
  uint32_t        numVertices = max(core->maxSrcVertex, core->maxDestVertex) + 1;
  uint32_t       *D;
  uint32_t        v;

  HML_ERR_GEN(count->D, cHmlErrGeneral);
  CALLOC(count->D, uint32_t, numVertices);
  /* use the just allocated count->D to store out-degree information */
  HML_ERR_PASS(hmlGraphCoreCountBidirectDegree(core, count->D, numVertices));
  /* create vertex permutation array: count->P */
  HML_ERR_GEN(count->P, cHmlErrGeneral);
  MALLOC(count->P, uint32_t, numVertices);
  for (v = 0; v < numVertices; v++) {
    count->P[v] = v;
  }
  HML_QSORT(count->P, numVertices, sizeof(uint32_t),
            hmlTriangleCountVertexCompare, count->D);
  CALLOC(D, uint32_t, numVertices);
  /* store the "adjusted" degree in 'D' and use it as
   * the 'D' argument to call hmlGraphCoreSetR(core, D, ...) */
  HML_ERR_PASS(hmlGraphCoreCountDegreeIfSmallerP(core, count->P,
               D, numVertices));
  /* make a shallow copy so that we can still free the memory allocated
   * for R and E in the original 'core'
   */
  memcpy(copy, core, sizeof(HmlGraphCore));
  /* since we've got a copy of 'core', we can (re)initialize 'core'
   * as if it never existed before
   */
  HML_ERR_PASS(hmlGraphCoreInit(core, numVertices, copy->numEdges));
  HML_ERR_PASS(hmlGraphCoreSetR(core, D, 0, numVertices - 1));
  insertState.P = count->P;
  HML_ERR_PASS(hmlGraphCoreInsertFromSrc(core,
                                         copy,
                                         hmlGraphCoreDefaultInsertIniter,
                                         hmlTriangleCountGraphInserter,
                                         NULL,
                                         &insertState));
  if (count->numThreads > 1 &&
      count->numThreads != count->partition.numPartitions) {
    HML_ERR_PASS(hmlTriangleCountSetPartition(count));
    HML_ERR_PASS(hmlGraphCoreSortEdgesByPartition(core, &count->partition));
  } else {
    HML_ERR_PASS(hmlGraphCoreSortEdges(core, 1));
  }
  /* free R and E in the old 'core' now that the new one has been created */
  HML_ERR_PASS(hmlGraphCoreDelete(copy));
  FREE(D);

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlTriangleCountBySearch(HmlGraphCore     *core,
                         uint32_t            thread,
                         uint32_t            minVertex,
                         uint32_t            maxVertex,
                         void             *args) {
  HML_ERR_PROLOGUE;
  HmlTriangleCountBase *count = (HmlTriangleCountBase *)args;
  uint32_t *R = core->R;
  uint32_t *E = core->E;
  uint32_t *eU;
  uint32_t *eU2;
  uint32_t *eV;
  uint32_t *endU;
  uint32_t *endV;
  uint32_t  u;
  uint32_t  v;
  uint32_t  w;
  int32_t   lowV;
  int32_t   midV;
  int32_t   highV;
  uint64_t  numTriangles = 0;

  minVertex = max(minVertex, core->minSrcVertex);
  maxVertex = min(maxVertex, core->maxSrcVertex);
  for(u = minVertex; u <= maxVertex; u++) {
    endU = &E[R[u + 1]] - 1;
    for(eU = &E[R[u]]; eU < endU; eU++) {
      v = *eU;
      /* due to lexicographic edge pruning, v may be > maxSrcVertex */
      if(v <= core->maxSrcVertex) {
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
        } else { /* use binary search */
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
            } else {
              lowV = 0;
              while(lowV <= highV) {
                /* to avoid overflow in (lowV + highV) / 2 */
                midV = lowV + (highV - lowV) / 2;
                if(eV[midV] == w) {
                  lowV = midV + 1;
                  numTriangles++;
                  break;
                } else if(eV[midV] < w) {
                  lowV = midV + 1;
                } else {
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
  count->numTrianglesEachThread[thread] = numTriangles;

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlTriangleCountByHash(HmlGraphCore     *core,
                       uint32_t            thread,
                       uint32_t            minVertex,
                       uint32_t            maxVertex,
                       void             *args) {
  HML_ERR_PROLOGUE;
  HmlTriangleCountBase *count = (HmlTriangleCountBase *)args;
  uint32_t *R = core->R;
  uint32_t *E = core->E;
  uint32_t  eU;
  uint32_t  eU2;
  uint32_t  eV;
  uint32_t  endU;
  uint32_t  u;
  uint32_t  v;
  uint64_t  numTriangles = 0;
  uint32_t *edgeId0;
  uint32_t *edgeId;
  uint32_t  numDestVertices = core->maxDestVertex - core->minDestVertex + 1;

  CALLOC(edgeId0, uint32_t, numDestVertices);
  /*  initialize edgeId0[] with (uint32_t)-1, an invalid edge id */
  memset(edgeId0, 0xFF, sizeof(uint32_t) * numDestVertices);
  edgeId = edgeId0 - core->minDestVertex;
  minVertex = max(minVertex, core->minSrcVertex);
  maxVertex = min(maxVertex, core->maxSrcVertex);
  for(u = minVertex; u <= maxVertex; u++) {
    endU = R[u + 1] - 1;
    for(eU = R[u]; eU < endU; eU++) {
      v = E[eU];
      /* due to lexicographic edge pruning, v may be > maxSrcVertex */
      if(v <= core->maxSrcVertex) {
        for(eV = R[v]; eV < R[v + 1]; eV++) {
          edgeId[E[eV]] = eU;
        }
        for(eU2 = eU + 1; eU2 <= endU; eU2++) {
          if(edgeId[E[eU2]] == eU) {
            numTriangles++;
          }
        }
      }
    }
  }
  count->numTrianglesEachThread[thread] = numTriangles;
  FREE(edgeId0);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTriangleCountRun(HmlTriangleCountBase            *count) {
  HML_ERR_PROLOGUE;
  HmlGraphCore        *core = &count->core;
  uint32_t               thread;
  HmlGraphCoreParaFunc func = (count->countByHash) ? hmlTriangleCountByHash :
                              hmlTriangleCountBySearch;

  if (count->numThreads > 1) {
    if (count->numThreads != count->partition.numPartitions) {
      HML_ERR_PASS(hmlTriangleCountSetPartition(count));
    }
    HML_ERR_PASS(hmlGraphCoreRunParaFuncByPartition(core, func, count,
                 &count->partition));
    count->numTriangles = 0;
    for (thread = 0; thread < count->numThreads; thread++) {
      count->numTriangles += count->numTrianglesEachThread[thread];
    }
  } else {
    count->numTrianglesEachThread = &count->numTriangles;
    HML_ERR_PASS(func(&count->core, 0,
                      count->core.minSrcVertex,
                      count->core.maxSrcVertex,
                      count));
    count->numTrianglesEachThread = NULL;
    /* count->numTriangles = count->numTrianglesEachThread[0]; */
  }

  HML_NORMAL_RETURN;
}
