/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include "hml_pagerank_spmv.h"
#include "hml_pagerank_spmv_kernel.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std;

#define cHmlBlocksPerGrid      128
#define cHmlThreadsPerBlock    128

#define cHmlPagerankSpmvPartitionSizeInit      1024
#define cHmlPagerankSpmvMaxNumPartitions       32

typedef pair<int, float> PagerankSpmvPair;

/* global variables defined here */
texture<uint32_t, 1> texDataR;

texture<uint32_t, 1> texDataE;

texture<float, 1> texDataMap0;

texture<float, 1> texDataMap1;

void
hmlGraphCopyHostToDevice(HmlGraph *devGraph, HmlGraph *hostGraph) {
  /* make a shallow copy of hostGraph in devGraph */
  memcpy(devGraph, hostGraph, sizeof(HmlGraph));

  /* allocate R[] on device */
  HANDLE_ERROR(cudaMalloc((void **)&devGraph->R,
                          sizeof(uint32_t) * hostGraph->sizeofR));
  /* allocate E[] on device */
  HANDLE_ERROR(cudaMalloc((void **)&devGraph->E,
                          sizeof(uint32_t) * hostGraph->sizeofE));

  /* copy host R[] to device R[] */
  HANDLE_ERROR(cudaMemcpy((void *)devGraph->R, (void *)hostGraph->R,
                          sizeof(uint32_t) * hostGraph->sizeofR,
                          cudaMemcpyHostToDevice));

  /* copy host E[] to device E[] */
  HANDLE_ERROR(cudaMemcpy((void *)devGraph->E, (void *)hostGraph->E,
                          sizeof(uint32_t) * hostGraph->sizeofE,
                          cudaMemcpyHostToDevice));
}

void
hmlGraphCopyHostToHost(HmlGraph *graphDest, HmlGraph *graphSrc) {
  /* make a shallow copy of graphSrc in graphDest */
  memcpy(graphDest, graphSrc, sizeof(HmlGraph));
  /* allocate R[] */
  MALLOC(graphDest->R, uint32_t, graphSrc->sizeofR);
  /* allocate E[] */
  MALLOC(graphDest->E, uint32_t, graphSrc->sizeofE);
  /* copy R[] from graphSrc to graphDest */
  memcpy(graphDest->R, graphSrc->R, sizeof(uint32_t) * graphSrc->sizeofR);
  /* copy E[] from graphSrc to graphDest */
  memcpy(graphDest->E, graphSrc->E, sizeof(uint32_t) * graphSrc->sizeofE);
}

void
hmlGraphDeleteHost(HmlGraph *graph) {
  FREE(graph->R);
  graph->sizeofR = 0;
  FREE(graph->E);
  graph->sizeofE = 0;
}

void
hmlGraphDeleteDevice(HmlGraph *graph) {
  HANDLE_ERROR(cudaFree(graph->R));
  HANDLE_ERROR(cudaFree(graph->E));
  graph->R = NULL;
  graph->sizeofR = 0;
  graph->E = NULL;
  graph->sizeofE = 0;
}

void
hmlGraphBindTexture(HmlGraph                    *graph,
                    const texture<uint32_t, 1> &texDataR,
                    const texture<uint32_t, 1> &texDataE) {
  size_t texOffset;

  /* bind row data to texture */
  HANDLE_ERROR(cudaBindTexture(&texOffset, texDataR, graph->R,
                               sizeof(uint32_t) * graph->sizeofR));
  /* check for non-zero offset */
  if(texOffset != 0) {
    fprintf(stderr, "; Error: Row texture offset != 0\n");
    exit(EXIT_FAILURE);
  }

  /* bind edge data to texture */
  HANDLE_ERROR(cudaBindTexture(&texOffset, texDataE, graph->E,
                               sizeof(uint32_t) * graph->sizeofE));
  /* check for non-zero offset */
  if(texOffset != 0) {
    fprintf(stderr, "; Error: Edge texture offset != 0\n");
    exit(EXIT_FAILURE);
  }
}

void
hmlGraphUnbindTexture(const texture<uint32_t, 1> &texDataR,
                      const texture<uint32_t, 1> &texDataE) {
  HANDLE_ERROR(cudaUnbindTexture(texDataR));
  HANDLE_ERROR(cudaUnbindTexture(texDataE));
}

bool
hmlPagerankSpmvPairComparator(const PagerankSpmvPair &p1, const PagerankSpmvPair &p2) {
  return p1.second > p2.second;
}

void
hmlPagerankSpmvPrintTopVertices(FILE    *file,
                                float *map,
                                uint32_t   numVertices,
                                uint32_t   printTopK) {
  float scoreSum = 0.0;
  printTopK = min(printTopK, numVertices);
  if(printTopK > 0) {
    vector<PagerankSpmvPair> pageRankPairVector;
    for(size_t i = 0; i < numVertices; ++i) {
      pageRankPairVector.push_back(make_pair(i, map[i]));
    }
    sort(pageRankPairVector.begin(), pageRankPairVector.end(),
         hmlPagerankSpmvPairComparator);
    fprintf(file, "; vertex_id=pagerank_score\n");
    for(size_t i = 0; i < printTopK; ++i) {
      fprintf(file, "%d=%8.6f\n", pageRankPairVector[i].first,
              pageRankPairVector[i].second);
      scoreSum += pageRankPairVector[i].second;
    }
    fprintf(stderr, "; Info: Top %d score sum = %f\n", printTopK, scoreSum);
  }
}

/* assumes map0 has been initialized, which is used as the very first
 * prevMap[]
 */
void hmlPagerankSpmvCpuIter(HmlGraph   *graph,
                            float  dampingFactor,
                            float *map0,
                            float *map1,
                            uint32_t   numIters) {
  //uint32_t   minSrcVertex = graph->minSrcVertex;
  //uint32_t   maxSrcVertex = graph->maxSrcVertex;
  uint32_t  *R = graph->R;
  uint32_t  *E = graph->E;
  uint32_t  *succ;
  uint32_t  *succMax;
  uint32_t   maxVertex = max(graph->maxSrcVertex, graph->maxDestVertex);
  float  oneMinusDoverN = (1.0 - dampingFactor) / (float)(maxVertex + 1);
  float  inputProbSum;
  float  pageRankScore;
  float *prevMap;
  float *curMap;
  float *tmpMap;

  prevMap = map0;
  curMap = map1;
  while(numIters--) {
    //for (uint32_t v = minSrcVertex; v <= maxSrcVertex; ++v) {
    for(uint32_t v = 0; v <= maxVertex; ++v) {
      succ = &E[R[v]];
      succMax = &E[R[v + 1]];
      pageRankScore = oneMinusDoverN;
      if(succ < succMax) {
        inputProbSum = 0.0;
        while(succ < succMax) {
          /* fprintf(file, "%d 0 %d %d\n", succ[0], succ[1], vid); */
          //inputProbSum += prevMap[succ[0]] * (*(float*)(&succ[1]));
          inputProbSum += prevMap[succ[0]] / succ[1];
          succ += 2;
        }
        pageRankScore += dampingFactor * inputProbSum;
      }
      curMap[v] = pageRankScore;
    }
    /* swap the maps */
    tmpMap = prevMap;
    prevMap = curMap;
    curMap = tmpMap;
  }
}

void
hmlPagerankSpmvCpu(HmlGraph   *hostGraph,
                   float       dampingFactor,
                   uint32_t    numIters,
                   uint32_t    printTopK,
                   const char *outFileNamePrefix,
                   const char *outFileNameExtension) {
  uint32_t   maxVertex = max(hostGraph->maxSrcVertex, hostGraph->maxDestVertex);
  uint32_t   numVertices = maxVertex + 1;
  uint32_t   v;
  float *map0;
  float *map1;
  double   cpuStart;
  double   cpuEnd;
  double   wallStart;
  double   wallEnd;
  FILE    *outFile;
  ostringstream outFilename;

  /* alloc memory for both float maps */
  MALLOC(map0, float, numVertices);
  MALLOC(map1, float, numVertices);
  /* init the first map */
  for(v = 0; v <= maxVertex; ++v) {
    map0[v] = (float) 1.0 / (float) numVertices;
  }
  /* iterate over the source vertices */
  hmlGetSecs(&cpuStart, &wallStart);
  hmlPagerankSpmvCpuIter(hostGraph, dampingFactor, map0, map1, numIters);
  hmlGetSecs(&cpuEnd, &wallEnd);
  fprintf(stderr, "; Info: CPU pagerank: cpu time = %.2lf, wall time = %.2lf\n",
          (cpuEnd - cpuStart) * 1000, (wallEnd - wallStart) * 1000);
  if(outFileNamePrefix) {
    outFilename << outFileNamePrefix << "(d=" << dampingFactor << ")."
                << outFileNameExtension;
    outFile = openFile(outFilename.str().c_str(), "wb");
  }
  else {
    outFile = stdout;
  }
  hmlPagerankSpmvPrintTopVertices(outFile, map0, numVertices, printTopK);
  if(outFile != stdout) {
    fclose(outFile);
  }
  FREE(map0);
  FREE(map1);
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
hmlPagerankSpmvPartitionVertexByOutDeg(HmlGraph *graph,
                                       uint32_t *minOutDeg,
                                       uint32_t  numPartitions,
                                       uint32_t *vertexRank,
                                       uint32_t *partitionPrefixSize) {
  uint32_t **partitions;
  uint32_t   p;
  uint32_t   v;
  uint32_t   outDeg;
  uint32_t  *R = graph->R;
  uint32_t  *pPartitionAllocSize;   /* allocation size */
  uint32_t **partitionPtr;
  uint32_t **partitionEndPtr;        /* actual used size */
  uint32_t   numSrcVertices = graph->maxSrcVertex - graph->minSrcVertex + 1;
  uint32_t   prefixSize = 0;

  MALLOC(partitions, uint32_t *, numPartitions);
  MALLOC(partitionPtr, uint32_t *, numPartitions);
  MALLOC(partitionEndPtr, uint32_t *, numPartitions);
  MALLOC(pPartitionAllocSize, uint32_t, numPartitions);
  for(p = 0; p < numPartitions; ++p) {
    MALLOC(partitions[p], uint32_t, cHmlPagerankSpmvPartitionSizeInit);
    pPartitionAllocSize[p] = cHmlPagerankSpmvPartitionSizeInit;
    partitionPtr[p] = partitions[p];
    partitionEndPtr[p] = partitions[p] + cHmlPagerankSpmvPartitionSizeInit;
  }
  for(v = graph->minSrcVertex; v <= graph->maxSrcVertex; ++v) {
    outDeg = (R[v + 1] - R[v]) / 2; /* each page takes two 32-bit words */
    /* use linear scan to find which partition this vertex belongs to */
    for(p = 0; p < numPartitions && minOutDeg[p] <= outDeg; ++p);
    if(p > 0) {
      --p;
      if(partitionPtr[p] == partitionEndPtr[p]) {
        REALLOC(partitions[p], uint32_t, pPartitionAllocSize[p] * 2);
        partitionPtr[p] = partitions[p] + pPartitionAllocSize[p];
        pPartitionAllocSize[p] *= 2;
        partitionEndPtr[p] = partitions[p] + pPartitionAllocSize[p];
      }
      *partitionPtr[p]++ = v;
    }
  }
  for(p = 0; p < numPartitions; ++p) {
    prefixSize += partitionPtr[p] - partitions[p];
    partitionPrefixSize[p] = prefixSize;
    if(prefixSize > numSrcVertices) {
      fprintf(stderr, "; Error: prefixSize = %d > numSrcVertices = %d\n",
              prefixSize, numSrcVertices);
      exit(EXIT_FAILURE);
    }
    memcpy((void *)vertexRank, (void *)partitions[p],
           sizeof(uint32_t) * (partitionPtr[p] - partitions[p]));
    vertexRank += partitionPtr[p] - partitions[p];
  }
  /* free memory */
  for(p = 0; p < numPartitions; ++p) {
    FREE(partitions[p]);
    FREE(partitionPtr);
    FREE(partitionEndPtr);
    FREE(pPartitionAllocSize);
  }
  FREE(partitions);
}

void
hmlPagerankSpmvGpu(HmlGraph   *hostGraph,
                   float       dampingFactor,
                   uint32_t    numIters,
                   uint32_t    printTopK,
                   const char *outFileNamePrefix,
                   const char *outFileNameExtension,
                   uint32_t    verbosity) {
  HmlGraph    devGraphVal;
  HmlGraph   *devGraph = &devGraphVal;
  uint32_t  *hostVertexRank; /* array of vertex ids sorted by out-degree */
  uint32_t  *devVertexRank; /* array of vertex ids sorted by out-degree */
  uint32_t   minSrcVertex = hostGraph->minSrcVertex;
  uint32_t   maxSrcVertex = hostGraph->maxSrcVertex;
  uint32_t   numSrcVertices = maxSrcVertex - minSrcVertex + 1;
  uint32_t   numVertices = max(maxSrcVertex, hostGraph->maxDestVertex) + 1;
  float  oneMinusDoverN = (1.0 - dampingFactor) / (float)numVertices;
  float *devMap0;
  float *devMap1;
  float *hostMap;
  float  initVal = (float) 1.0 / (float) numVertices;
  uint32_t   minOutDeg[cHmlPagerankSpmvMaxNumPartitions]; /* min out-deg of each partition */
  uint32_t   partitionPrefixSize[cHmlPagerankSpmvMaxNumPartitions]; /* cumulative size */
  uint32_t   vertexofRankSize;
  uint32_t   numPartitions;
  double   cpuStart;
  double   cpuEnd;
  double   wallStart;
  double   wallEnd;
  size_t   freeBytesStart;
  size_t   totalBytesStart;
  size_t   freeBytesEnd;
  size_t   totalBytesEnd;
  bool     useTextureMem;
  PagerankSpmvKernel kernelEven[cHmlPagerankSpmvMaxNumKernels];
  PagerankSpmvKernel kernelOdd[cHmlPagerankSpmvMaxNumKernels];
  HmlPagerankSpmvKernelArg      kernelArg[cHmlPagerankSpmvMaxNumPartitions];
  FILE         *outFile;
  ostringstream outFilename;

  /* get free gpu memory size */
  if(verbosity >= 2) {
    HANDLE_ERROR(cudaMemGetInfo(&freeBytesStart, &totalBytesStart));
    fprintf(stderr, "; Info: GPU memory: %ld bytes free, %ld bytes total\n",
            freeBytesStart, totalBytesStart);
  }
  /* create device graph */
  hmlGetSecs(&cpuStart, &wallStart);
  hmlGraphCopyHostToDevice(devGraph, hostGraph);
  cudaDeviceSynchronize();
  if(verbosity >= 2) {
    hmlGetSecs(&cpuEnd, &wallEnd);
    fprintf(stderr, "; Info: Load graph to device: "
            "cpu time = %.2lf, wall time = %.2lf\n",
            (cpuEnd - cpuStart) * 1000, (wallEnd - wallStart) * 1000);
  }

  /* do NOT use texture memory for R and E */
  useTextureMem = false; /* false (default) is faster */
  if(numVertices > cHmlMaxCudaTexture1DLinear) {
    fprintf(stderr, "; Error: Number of vertices exceeds the maximum "
            "texture 1D size\n");
    exit(EXIT_FAILURE);
  }
  if(useTextureMem && devGraph->sizeofR <= cHmlMaxCudaTexture1DLinear
      && devGraph->sizeofE <= cHmlMaxCudaTexture1DLinear) {
    hmlGraphBindTexture(devGraph, texDataR, texDataE);
  }
  else {
    useTextureMem = false;
  }

  hmlPagerankSpmvKernelSetup(kernelEven, kernelOdd, minOutDeg,
                             cHmlPagerankSpmvMaxNumKernels, useTextureMem,
                             &numPartitions, kernelArg);

  /* create vertexofRank mapping */
  hmlGetSecs(&cpuStart, &wallStart);
  /* allocate hostVertexRank[] on CPU */
  MALLOC(hostVertexRank, uint32_t, numSrcVertices);
  hmlPagerankSpmvPartitionVertexByOutDeg(hostGraph, minOutDeg, numPartitions,
                                         hostVertexRank, partitionPrefixSize);
  //hmlGetSecs(&cpuEnd, &wallEnd);
  //fprintf(stderr, "; Info: Partition vertices on CPU: "
  //      "cpu time = %.2lf, wall time = %.2lf\n",
  //      (cpuEnd - cpuStart* 1000, (wallEnd - wallStart) * 1000);

  vertexofRankSize = partitionPrefixSize[numPartitions - 1];
  /* resize hostVertexRank */
  REALLOC(hostVertexRank, uint32_t, vertexofRankSize);
  /* allocate devVertexRank[] on device */
  HANDLE_ERROR(cudaMalloc((void **)&devVertexRank,
                          sizeof(uint32_t) * vertexofRankSize));
  /* copy hostVertexRank[] to devVertexRank[] */
  HANDLE_ERROR(cudaMemcpy((void *)devVertexRank, (void *)hostVertexRank,
                          sizeof(uint32_t) * vertexofRankSize,
                          cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  if(verbosity >= 2) {
    hmlGetSecs(&cpuEnd, &wallEnd);
    fprintf(stderr, "; Info: Partition and copy vertice ranks to device: "
            "cpu time = %.2lf, wall time = %.2lf\n",
            (cpuEnd - cpuStart) * 1000, (wallEnd - wallStart) * 1000);
    fprintf(stderr, "; Info: Number of pages with in-coming link: "
            "%d (%.2lf%%)\n",
            vertexofRankSize, 100 * vertexofRankSize/(double)(numVertices));
    fprintf(stderr, "; Info: Partitioned graph size = %.2lf MB\n",
            (hostGraph->sizeofR + hostGraph->sizeofE + vertexofRankSize) *
            sizeof(uint32_t) / (double)(1024 * 1024));
  }
  /* print vertex ranks for small graphs */
  if(verbosity >= 3 && vertexofRankSize <= 100) {
    for(uint32_t r = 0; r < vertexofRankSize; ++r) {
      fprintf(stderr, "; Info: rank %3d = vertex %3d\n",
              r, hostVertexRank[r]);
    }
  }

  /* set the kernel arguments */
  hmlPagerankSpmvKernelArgSet(kernelArg, numPartitions, minOutDeg, hostVertexRank,
                              hostGraph->R, partitionPrefixSize);

  /* print kernel params */
  if(verbosity >= 2) {
    hmlPagerankSpmvKernelArgPrint(kernelArg, numPartitions);
  }

  hmlGetSecs(&cpuStart, &wallStart);

  devMap0 = hmlDeviceFloatArrayAllocBind(numVertices, texDataMap0);
  devMap1 = hmlDeviceFloatArrayAllocBind(numVertices, texDataMap1);

  hmlPagerankSpmvInitKernel<<<cHmlBlocksPerGrid, cHmlThreadsPerBlock>>>
  (devMap0, initVal, numVertices);

  for(uint32_t iter = 0; iter < numIters; ++iter) {
    //fprintf(stderr, "; Info: iter = %d\n", iter);
    if(iter % 2 == 0) {
      hmlPagerankSpmvInitKernel<<<cHmlBlocksPerGrid, cHmlThreadsPerBlock>>>
      (devMap1, oneMinusDoverN, numVertices);
      for(uint32_t p = 0; p < numPartitions; ++p) {
        kernelEven[kernelArg[p].id]<<<kernelArg[p].grid, kernelArg[p].block>>>
        (devMap1, devGraph->R, devGraph->E, devVertexRank,
         kernelArg[p].minVertexRank, kernelArg[p].maxVertexRank,
         dampingFactor);
      }
    }
    else {
      hmlPagerankSpmvInitKernel<<<cHmlBlocksPerGrid, cHmlThreadsPerBlock>>>
      (devMap0, oneMinusDoverN, numVertices);
      for(uint32_t p = 0; p < numPartitions; ++p) {
        kernelOdd[kernelArg[p].id]<<<kernelArg[p].grid, kernelArg[p].block>>>
        (devMap0, devGraph->R, devGraph->E, devVertexRank,
         kernelArg[p].minVertexRank, kernelArg[p].maxVertexRank,
         dampingFactor);
      }
    }
  }
  cudaDeviceSynchronize();
  if(verbosity >= 1) {
    hmlGetSecs(&cpuEnd, &wallEnd);
    fprintf(stderr, "; Info: GPU pagerank: cpu time = %.2lf, wall time = %.2lf\n",
            (cpuEnd - cpuStart) * 1000, (wallEnd - wallStart) * 1000);
  }
  if(verbosity >= 2) {
    HANDLE_ERROR(cudaMemGetInfo(&freeBytesEnd, &totalBytesEnd));
    fprintf(stderr, "; Info: GPU memory: %ld bytes free, %ld bytes total\n",
            freeBytesStart, totalBytesStart);
    fprintf(stderr, "; Info: GPU memory used by pagerank: %ld bytes\n",
            freeBytesStart - freeBytesEnd);
  }
  if(verbosity >= 1) {
    hmlGetSecs(&cpuStart, &wallStart);
  }
  MALLOC(hostMap, float, numVertices);
  HANDLE_ERROR(cudaMemcpy((void *)hostMap, (void *)devMap0,
                          sizeof(float) * numVertices,
                          cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaUnbindTexture(texDataMap0));
  HANDLE_ERROR(cudaUnbindTexture(texDataMap1));
  HANDLE_ERROR(cudaFree(devMap0));
  HANDLE_ERROR(cudaFree(devMap1));
  printTopK = min(printTopK, numVertices);
  if(outFileNamePrefix) {
    outFilename << outFileNamePrefix << "(d=" << dampingFactor << ")."
                << outFileNameExtension;
    outFile = openFile(outFilename.str().c_str(), "wb");
  }
  else {
    outFile = stdout;
  }
  hmlPagerankSpmvPrintTopVertices(outFile, hostMap, numVertices, printTopK);
  if(outFile != stdout) {
    fclose(outFile);
  }

  if(verbosity >= 1) {
    hmlGetSecs(&cpuEnd, &wallEnd);
    fprintf(stderr, "; Info: CPU sort top %d pages: "
            "cpu time = %.2lf, wall time = %.2lf\n", printTopK,
            (cpuEnd - cpuStart) * 1000, (wallEnd - wallStart) * 1000);
  }
  FREE(hostMap);
  FREE(hostVertexRank);
  if(useTextureMem) {
    hmlGraphUnbindTexture(texDataR, texDataE);
  }
  hmlGraphDeleteDevice(devGraph);
  HANDLE_ERROR(cudaFree(devVertexRank));
}
