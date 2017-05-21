/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_pagerank.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "hml_tsv2_utils.h"
#include "hml_file_utils.h"

using namespace std;

#define cHmlBlocksPerGrid      128
#define cHmlThreadsPerBlock    128

#define cHmlPagerankPartitionSizeInit      1024
#define cHmlPagerankMaxNumPartitions       32

typedef pair<int, float> PagerankPair;

/* global variables defined here */
/* WARNING: Do NOT change the names of these texture variables */
/* Make sure the same names are used in hml_...kernel_template.h */
texture<uint32_t, 1> texDataR;

texture<uint32_t, 1> texDataE;

texture<uint32_t, 1> texDataD;

texture<float, 1> texDataVec0;

texture<float, 1> texDataVec1;

bool
hmlPagerankPairComparator(const PagerankPair &p1, const PagerankPair &p2) {
  return p1.second > p2.second;
}

void
hmlPagerankPrintTopVertices(FILE      *file,
                            float     *map,
                            uint32_t   numVertices,
                            uint32_t   printTopK) {
  float scoreSum = 0.0;
  printTopK = min(printTopK, numVertices);
  if (printTopK > 0) {
    vector<PagerankPair> pageRankPairVector;
    for (size_t i = 0; i < numVertices; ++i) {
      pageRankPairVector.push_back(make_pair(i, map[i]));
    }
    sort(pageRankPairVector.begin(), pageRankPairVector.end(),
      hmlPagerankPairComparator);
    fprintf(file, "; vertex_id=pagerank_score\n");
    for (size_t i = 0; i < printTopK; ++i) {
      fprintf(file, "%d=%8.6f\n", pageRankPairVector[i].first,
        pageRankPairVector[i].second);
      scoreSum += pageRankPairVector[i].second;
    }
    fprintf(stderr, "; Info: Top %d score sum = %f\n", printTopK, scoreSum);
  }
}

static HmlErrCode
hmlPagerankInitVectors(HmlPagerank *pagerank) {
  HML_ERR_PROLOGUE;
  uint32_t   v;
  float *vector0;
  uint32_t   numVertices = max(pagerank->core.maxSrcVertex,
                             pagerank->core.maxDestVertex) + 1;

  if(!pagerank->vector0) {
    MALLOC(pagerank->vector0, float, numVertices);
  }
  if(!pagerank->vector1) {
    MALLOC(pagerank->vector1, float, numVertices);
  }
  vector0 = pagerank->vector0;
  for(v = 0; v < numVertices; ++v) {
    vector0[v] = 1.0f / (float)numVertices;
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPagerankCpu(HmlPagerank *pagerank) {
  HML_ERR_PROLOGUE;
  float  dampingFactor = pagerank->dampingFactor;
  uint32_t   numIters = pagerank->numIters;
  uint32_t  *outDegreeArr = pagerank->outDegreeArr;
  uint32_t  *R = pagerank->core.R;
  uint32_t  *E = pagerank->core.E;
  uint32_t  *e;
  uint32_t   v;
  uint32_t   maxNumSrcVertices = pagerank->core.maxNumSrcVertices;
  uint32_t   numVertices = max(pagerank->core.maxSrcVertex,
                             pagerank->core.maxDestVertex) + 1;
  float  oneMinusDoverN = (1.0f - dampingFactor) / (float)numVertices;
  float  inputProbSum;
  float *vectorPre;
  float *vectorNow;
  float *vectorTmp;

  hmlPagerankInitVectors(pagerank);
  vectorPre = pagerank->vector0;
  vectorNow = pagerank->vector1;
  while(numIters--) {
    for(v = 0; v < maxNumSrcVertices; ++v) {
      inputProbSum = 0.0;
      for(e = &E[R[v]]; e < &E[R[v + 1]]; ++e) {
        inputProbSum += vectorPre[*e] / outDegreeArr[*e];
      }
      vectorNow[v] = oneMinusDoverN + dampingFactor * inputProbSum;
    }
    /* swap the vectors */
    vectorTmp = vectorPre;
    vectorPre = vectorNow;
    vectorNow = vectorTmp;
  }

  HML_NORMAL_RETURN;
}

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
hmlPagerankPartitionVertexByOutDeg(HmlGraphCore  *core,
                                   uint32_t *minOutDeg,
                                   uint32_t  numPartitions,
                                   uint32_t *vertexRank,
                                   uint32_t *partitionPrefixSize) {
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
    MALLOC(partitions[p], uint32_t, cHmlPagerankPartitionSizeInit);
    pPartitionAllocSize[p] = cHmlPagerankPartitionSizeInit;
    partitionPtr[p] = partitions[p];
    partitionEndPtr[p] = partitions[p] + cHmlPagerankPartitionSizeInit;
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

HmlErrCode
hmlPagerankGpuInit(HmlPagerank *pagerank) {
  HML_ERR_PROLOGUE;
  HmlGraphCore *core = &pagerank->core;
  uint32_t   minSrcVertex = core->minSrcVertex;
  uint32_t   maxSrcVertex = core->maxSrcVertex;
  uint32_t   numSrcVertices = maxSrcVertex - minSrcVertex + 1;
  uint32_t   numVertices = max(maxSrcVertex, core->maxDestVertex) + 1;
  uint32_t  *vertexRank; /* array of vertex ids sorted by out-degree */
  uint32_t   minOutDeg[cHmlPagerankMaxNumPartitions]; /* min out-deg of each partition */
  uint32_t   partitionPrefixSize[cHmlPagerankMaxNumPartitions]; /* cumulative size */
  uint32_t   vertexRankSize;
  size_t   freeBytes;
  size_t   totalBytes;
  double   cpuStart;
  double   cpuEnd;
  double   wallStart;
  double   wallEnd;

  if (!pagerank->vector0) {
    MALLOC(pagerank->vector0, float, numVertices);
  }
  /* get free gpu memory size */
  if (pagerank->verbosity >= 2) {
    HANDLE_ERROR(cudaMemGetInfo(&freeBytes, &totalBytes));
    fprintf(stderr, "; Info: GPU memory: %ld bytes free, %ld bytes total\n",
            freeBytes, totalBytes);
  }
  /* create device graph */
  hmlGetSecs(&cpuStart, &wallStart);
  hmlGraphCoreCopyToGpu(&pagerank->core, &pagerank->gpuCore);
  cudaDeviceSynchronize();
  hmlGetSecs(&cpuEnd, &wallEnd);
  if (pagerank->verbosity >= 2) {
    fprintf(stderr, "; Info: Load graph to device: wall time = %.2lf\n",
            (wallEnd - wallStart) * 1000);
  }
  if (numVertices > cHmlMaxCudaTexture1DLinear) {
    hmlPrintf("; Error: Number of vertices exceeds the maximum "
              "texture 1D size\n");
    HML_ERR_GEN(true, cHmlErrGeneral);
  }
  if (pagerank->useTextureMem && pagerank->gpuCore.maxNumSrcVertices
      <= cHmlMaxCudaTexture1DLinear
      && core->numEdges <= cHmlMaxCudaTexture1DLinear) {
    hmlGraphCoreBindTexture(&pagerank->gpuCore, texDataR, texDataE);
  }
  else {
    pagerank->useTextureMem = false;
  }
  hmlPagerankKernelSetup(pagerank->kernelEven, pagerank->kernelOdd, minOutDeg,
                         cHmlPagerankMaxNumKernels, pagerank->useTextureMem,
                         &pagerank->numPartitions, pagerank->kernelArgs);

  /* create vertexRank mapping */
  hmlGetSecs(&cpuStart, &wallStart);
  /* allocate vertexRank[] on CPU */
  MALLOC(vertexRank, uint32_t, numSrcVertices);
  hmlPagerankPartitionVertexByOutDeg(&pagerank->core, minOutDeg,
                                     pagerank->numPartitions,
                                     vertexRank, partitionPrefixSize);
  //hmlGetSecs(&cpuEnd, &wallEnd);
  //fprintf(stderr, "; Info: Partition vertices on CPU: "
  //      "cpu time = %.2lf, wall time = %.2lf\n",
  //      (cpuEnd - cpuStart* 1000, (wallEnd - wallStart) * 1000);

  vertexRankSize = partitionPrefixSize[pagerank->numPartitions - 1];
  /* resize vertexRank */
  REALLOC(vertexRank, uint32_t, vertexRankSize);
  /* allocate gpuVertexRank[] on device */
  HANDLE_ERROR(cudaMalloc(&pagerank->gpuVertexRank,
                          sizeof(uint32_t) * vertexRankSize));
  /* copy vertexRank[] to gpuVertexRank[] */
  HANDLE_ERROR(cudaMemcpy(pagerank->gpuVertexRank,
                          vertexRank,
                          sizeof(uint32_t) * vertexRankSize,
                          cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  hmlGetSecs(&cpuEnd, &wallEnd);
  if (pagerank->verbosity >= 2) {
    fprintf(stderr, "; Info: Partition and copy vertice ranks to device: "
            "wall time = %.2lf\n", (wallEnd - wallStart) * 1000);
    fprintf(stderr, "; Info: Number of pages with in-coming link: %d (%.2lf%%)\n",
            vertexRankSize, 100 * vertexRankSize/(double)(numVertices));
    fprintf(stderr, "; Info: Partitioned graph size = %.2lf MB\n",
            (core->maxNumSrcVertices + core->numEdges + vertexRankSize) *
            sizeof(uint32_t) / (double)(1024 * 1024));
  }
  /* print vertex ranks for small graphs */
  if (pagerank->verbosity >= 3 && vertexRankSize <= 100) {
    for (uint32_t r = 0; r < vertexRankSize; ++r) {
      fprintf(stderr, "; Info: rank %3d = vertex %3d\n", r, vertexRank[r]);
    }
  }

  /* set the kernel arguments */
  hmlPagerankKernelArgSet(pagerank->kernelArgs, pagerank->numPartitions,
                          minOutDeg, partitionPrefixSize);

  /* print kernel params */
  if (pagerank->verbosity >= 2)
    hmlPagerankKernelArgPrint(pagerank->kernelArgs, pagerank->numPartitions);

  pagerank->gpuVector0 = hmlDeviceFloatArrayAllocBind(numVertices, texDataVec0);
  pagerank->gpuVector1 = hmlDeviceFloatArrayAllocBind(numVertices, texDataVec1);
  pagerank->gpuOutDegreeArr =
    hmlDeviceUint32ArrayAllocBind(pagerank->outDegreeArrSize, texDataD);
  /* copy outDegreeArr[] from cpu to gpu */
  HANDLE_ERROR(cudaMemcpy(pagerank->gpuOutDegreeArr,
                          pagerank->outDegreeArr,
                          sizeof(uint32_t) * pagerank->outDegreeArrSize,
                          cudaMemcpyHostToDevice));
  FREE(vertexRank);
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPagerankGpu(HmlPagerank *pagerank) {
  HML_ERR_PROLOGUE;
  HmlGraphCore   *gpuCore = &pagerank->gpuCore;
  uint32_t  *gpuVertexRank = pagerank->gpuVertexRank;
  float  dampingFactor = pagerank->dampingFactor;
  uint32_t   numVertices = max(gpuCore->maxSrcVertex, gpuCore->maxDestVertex) + 1;
  float  oneMinusDoverN = (1.0 - dampingFactor) / (float)numVertices;
  float *gpuVector0 = pagerank->gpuVector0;
  float *gpuVector1 = pagerank->gpuVector1;
  float  initVal = (float) 1.0 / (float) numVertices;
  uint32_t   numPartitions = pagerank->numPartitions;
  double   cpuStart;
  double   cpuEnd;
  double   wallStart;
  double   wallEnd;
  HmlPagerankKernel    *kernelEven = pagerank->kernelEven;
  HmlPagerankKernel    *kernelOdd  = pagerank->kernelOdd;
  HmlPagerankKernelArg *kernelArgs = pagerank->kernelArgs;

  hmlGetSecs(&cpuStart, &wallStart);
  hmlPagerankInitKernel<<<cHmlBlocksPerGrid, cHmlThreadsPerBlock>>>
    (gpuVector0, initVal, numVertices);

  for (uint32_t iter = 0; iter < pagerank->numIters; ++iter) {
    //fprintf(stderr, "; Info: iter = %d\n", iter);
    if (iter % 2 == 0) {
      hmlPagerankInitKernel<<<cHmlBlocksPerGrid, cHmlThreadsPerBlock>>>
        (gpuVector1, oneMinusDoverN, numVertices);
      for (uint32_t p = 0; p < numPartitions; ++p) {
        kernelEven[kernelArgs[p].id]<<<kernelArgs[p].grid, kernelArgs[p].block>>>
          (gpuVector1, gpuCore->R, gpuCore->E, gpuVertexRank,
           kernelArgs[p].minVertexRank, kernelArgs[p].maxVertexRank,
           dampingFactor);
      }
    }
    else {
      hmlPagerankInitKernel<<<cHmlBlocksPerGrid, cHmlThreadsPerBlock>>>
        (gpuVector0, oneMinusDoverN, numVertices);
      for (uint32_t p = 0; p < numPartitions; ++p) {
        kernelOdd[kernelArgs[p].id]<<<kernelArgs[p].grid, kernelArgs[p].block>>>
          (gpuVector0, gpuCore->R, gpuCore->E, gpuVertexRank,
           kernelArgs[p].minVertexRank, kernelArgs[p].maxVertexRank,
           dampingFactor);
      }
    }
  }
  cudaDeviceSynchronize();
  hmlGetSecs(&cpuEnd, &wallEnd);
  if (pagerank->verbosity >= 1) {
    fprintf(stderr, "; Info: GPU pagerank: wall time = %.2lf\n",
            (wallEnd - wallStart) * 1000);
  }
  HANDLE_ERROR(cudaMemcpy(pagerank->vector0, gpuVector0,
                          sizeof(float) * numVertices,
                          cudaMemcpyDeviceToHost));

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPagerankInit(HmlPagerank *pagerank) {
  HML_ERR_PROLOGUE;
  memset(pagerank, 0, sizeof(HmlPagerank));
  pagerank->dampingFactor = cHmlPagerankDampingFactorDefault;
  pagerank->topK = 100;
  /* do NOT use texture memory for R and E */
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPagerankDelete(HmlPagerank *pagerank) {
  HML_ERR_PROLOGUE;

  FREE(pagerank->vector0);
  FREE(pagerank->vector1);
  FREE(pagerank->topKVertexValue);
  hmlGraphCoreDelete(&pagerank->core);
  /* free GPU stuff */
  HANDLE_ERROR(cudaFree(pagerank->gpuVector0));
  HANDLE_ERROR(cudaFree(pagerank->gpuVector1));
  HANDLE_ERROR(cudaFree(pagerank->gpuOutDegreeArr));
  if (pagerank->gpuCore.numEdges > 0) {
    hmlGraphCoreGpuDelete(&pagerank->gpuCore);
  }
  HANDLE_ERROR(cudaFree(pagerank->gpuVertexRank));
  HANDLE_ERROR(cudaUnbindTexture(texDataVec0));
  HANDLE_ERROR(cudaUnbindTexture(texDataVec1));
  HANDLE_ERROR(cudaUnbindTexture(texDataD));
  if (pagerank->useTextureMem) {
    hmlGraphCoreUnbindTexture(texDataR, texDataE);
  }

  memset(pagerank, 0, sizeof(HmlPagerank));

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlPagerankReadDegreeFile(HmlPagerank *pagerank,
                          FILE        *file) {
  HML_ERR_PROLOGUE;

  hmlTsv2InOutDegreeReadFile(file,
                             &pagerank->inDegreeArr,
                             &pagerank->inDegreeArrSize,
                             &pagerank->outDegreeArr,
                             &pagerank->outDegreeArrSize,
                             &pagerank->numEdges);

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlPagerankDegreeCountFile(HmlPagerank *pagerank,
                           FILE        *file) {
  HML_ERR_PROLOGUE;

  hmlTsv2InOutDegreeCountFile(file,
                              &pagerank->inDegreeArr,
                              &pagerank->inDegreeArrSize,
                              &pagerank->outDegreeArr,
                              &pagerank->outDegreeArrSize,
                              &pagerank->numEdges);
  rewind(file);
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPagerankSetInputFiles(HmlPagerank *pagerank,
                         FILE        *graphFile,
                         FILE        *inOutDegreeFile) {
  HML_ERR_PROLOGUE;

  if(inOutDegreeFile) {
    hmlPagerankReadDegreeFile(pagerank, inOutDegreeFile);
  }
  else {
    hmlPagerankDegreeCountFile(pagerank, graphFile);
  }
  pagerank->graphFile = graphFile;
  pagerank->inOutDegreeFile = inOutDegreeFile;
  pagerank->maxNumSrcVertices = pagerank->inDegreeArrSize;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPagerankReadTsv2InFile(HmlPagerank *pagerank) {
  HML_ERR_PROLOGUE;

  hmlGraphCoreInit(&pagerank->core,
                   pagerank->maxNumSrcVertices,
                   pagerank->numEdges);
  hmlGraphCoreSetR(&pagerank->core,
                   pagerank->inDegreeArr,
                   0,
                   pagerank->maxNumSrcVertices - 1);
  hmlGraphCoreReadTsv2File(&pagerank->core,
                           pagerank->graphFile, true);
  /* once the tsv2 file is read, pagerank->core.D is useless */
  hmlGraphCoreDeleteD(&pagerank->core);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPagerankFindTopK(HmlPagerank *pagerank) {
  HML_ERR_PROLOGUE;
  uint32_t v;
  uint32_t u;
  float *vector = pagerank->vector0;
  float  minTopKValue;
  uint32_t topK = pagerank->topK;
  uint32_t numVertices = max(pagerank->core.maxSrcVertex,
                           pagerank->core.maxDestVertex) + 1;
  HmlVertexfloat *topKVertexValue;

  HML_ERR_GEN(topK == 0, cHmlErrGeneral);
  HML_ERR_GEN(topK > numVertices, cHmlErrGeneral);
  if(!pagerank->topKVertexValue) {
    MALLOC(pagerank->topKVertexValue, HmlVertexfloat, topK);
  }
  topKVertexValue = pagerank->topKVertexValue;
  /* insert the first vertex-value pair */
  topKVertexValue[0].vertex = 0;
  topKVertexValue[0].value = vector[0];
  for(v = 1; v < topK; ++v) {
    for(u = v; u > 0 && topKVertexValue[u - 1].value < vector[v]; --u) {
      topKVertexValue[u].value = topKVertexValue[u - 1].value;
      topKVertexValue[u].vertex = topKVertexValue[u - 1].vertex;
    }
    topKVertexValue[u].value = vector[v];
    topKVertexValue[u].vertex = v;
  }
  minTopKValue = topKVertexValue[topK - 1].value;
  for(v = topK; v < numVertices; ++v) {
    if(minTopKValue >= vector[v]) {
      continue;
    }
    for(u = topK - 1; u > 0 && topKVertexValue[u - 1].value < vector[v]; --u) {
      topKVertexValue[u].value = topKVertexValue[u - 1].value;
      topKVertexValue[u].vertex = topKVertexValue[u - 1].vertex;
    }
    topKVertexValue[u].value = vector[v];
    topKVertexValue[u].vertex = v;
    minTopKValue = topKVertexValue[topK - 1].value;
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPagerankPrintTopK(HmlPagerank *pagerank, FILE *file) {
  HML_ERR_PROLOGUE;
  uint32_t v;
  HmlVertexfloat *topKVertexValue = pagerank->topKVertexValue;

  if(file == stdout) {
    hmlFileSetBinaryMode(file);
  }
  fprintf(file, "; vertex_id=pagerank_score\n");
  for(v = 0; v < pagerank->topK; ++v) {
    fprintf(file, "%u=%8.6f\n", topKVertexValue[v].vertex,
            topKVertexValue[v].value);
  }

  HML_NORMAL_RETURN;
}
