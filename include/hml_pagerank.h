/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_PAGERANK_H_INCLUDED_
#define HML_PAGERANK_H_INCLUDED_

#include "hml_common.h"
#include "hml_graph_core.h"
#include "hml_pagerank_kernel.h"

#define cHmlPagerankDampingFactorDefault   0.85f

typedef struct {
  uint32_t  vertex;
  float value;
} HmlVertexfloat;

typedef struct {
  uint32_t                   verbosity;
  FILE                    *graphFile;
  FILE                    *inOutDegreeFile;
  HmlGraphCore             core;
  uint32_t                  *inDegreeArr;
  uint32_t                   inDegreeArrSize;
  uint32_t                  *outDegreeArr;
  uint32_t                   outDegreeArrSize;
  uint32_t                   maxNumSrcVertices;
  uint64_t                   numEdges;
  uint32_t                   numIters;
  float                  dampingFactor;
  float                 *vector0;
  float                 *vector1;
  HmlVertexfloat        *topKVertexValue;
  uint32_t                   topK;

  /* GPU stuff */
  bool                     useTextureMem;
  HmlPagerankKernelArg     kernelArgs[cHmlPagerankMaxNumKernels];
  HmlPagerankKernel        kernelEven[cHmlPagerankMaxNumKernels];
  HmlPagerankKernel        kernelOdd[cHmlPagerankMaxNumKernels];
  HmlGraphCore             gpuCore;
  uint32_t                   numPartitions;
  /* array of vertex ids sorted by out-degree */
  uint32_t                  *gpuVertexRank;
  float                 *gpuVector0;
  float                 *gpuVector1;
  uint32_t                  *gpuOutDegreeArr;
} HmlPagerank;

void
hmlGraphReadTsv4(char *filename, bool sortedBySubject, HmlGraph *graph);

void
hmlGraphPrintStats(FILE *file, HmlGraph *graph);

void
hmlGraphPrintEdges(FILE *file, HmlGraph *graph, bool sortedBySubject);

void
hmlGraphDeleteHost(HmlGraph *graph);

HmlErrCode
hmlPagerankInit(HmlPagerank *pagerank);

HmlErrCode
hmlPagerankDelete(HmlPagerank *pagerank);

HmlErrCode
hmlPagerankSetInputFiles(HmlPagerank *pagerank,
                         FILE        *graphFile,
                         FILE        *inOutDegreeFile);

HmlErrCode
hmlPagerankReadTsv2InFile(HmlPagerank *pagerank);

HmlErrCode
hmlPagerankGpuInit(HmlPagerank *pagerank);

HmlErrCode
hmlPagerankFindTopK(HmlPagerank *pagerank);

HmlErrCode
hmlPagerankPrintTopK(HmlPagerank *pagerank, FILE *file);

HmlErrCode
hmlPagerankGpu(HmlPagerank *pagerank);

HmlErrCode
hmlPagerankCpu(HmlPagerank *pagerank);

#endif /* HML_PAGERANK_H_INCLUDED_ */
