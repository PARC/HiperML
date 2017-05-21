/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#ifndef HML_TRIANGLE_COUNT_H_INCLUDED_
#define HML_TRIANGLE_COUNT_H_INCLUDED_

#include "hml_common.h"
#include "hml_graph_core.h"
#include "hml_vertex_partition.h"
#include "hml_triangle_count_kernel.h"

typedef struct {
    HmlGraphCore       core;
    HmlVertexPartition partition;
    uint32_t          *D;  /* bi-directional degree of vertices */
    uint32_t          *P;  /* vertex permutation array */
    bool               countByHash;
    uint32_t           numThreads;
    uint64_t           numTriangles;
    uint64_t          *numTrianglesEachThread;
} HmlTriangleCountBase;

typedef struct {
  HmlTriangleCountBase       cpu;
  HmlTriangleCountBase       gpu;
  uint32_t                   verbosity;
  uint64_t                  *blockCountArr;

  /* GPU stuff */
  HmlTriangleCountKernelArg  kernelArgs[cHmlTriangleCountMaxNumKernels];
  HmlTriangleCountKernel     kernel[cHmlTriangleCountMaxNumKernels];
  uint32_t                   numPartitions;
  /* array of vertex ids sorted by out-degree */
  uint32_t                  *gpuVertexRank;
  uint32_t                  *gpuCountArr;
  uint64_t                  *gpuBlockCountArr;
} HmlTriangleCount;

HmlErrCode
hmlTriangleCountInit(HmlTriangleCount *triangleCount);

HmlErrCode
hmlTriangleCountDelete(HmlTriangleCount *triangleCount);

HmlErrCode
hmlTriangleCountSetInputFiles(HmlTriangleCount *triangleCount,
                         FILE        *graphFile,
                         FILE        *inOutDegreeFile);

HmlErrCode
hmlTriangleCountReadTsv2InFile(HmlTriangleCount *triangleCount);

HmlErrCode
hmlTriangleCountReadOrderedTsv2File(HmlTriangleCountBase *count,
                                    FILE *file,
                                    bool srcVertexOnRightColumn);

HmlErrCode
hmlTriangleCountReadOrderedTsv2FileByName(HmlTriangleCountBase *count,
                                          char const *fileName,
                                          bool srcVertexOnRightColumn);

HmlErrCode
hmlTriangleCountGpuInit(HmlTriangleCount *triangleCount);

HmlErrCode
hmlTriangleCountGpu(HmlTriangleCount *triangleCount);

HmlErrCode
hmlTriangleCountRun(HmlTriangleCountBase *count);

#endif /* HML_TRIANGLE_COUNT_H_INCLUDED_ */
