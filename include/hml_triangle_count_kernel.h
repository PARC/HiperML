/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#ifndef HML_TRIANGLE_COUNT_KERNEL_H_INCLUDED_
#define HML_TRIANGLE_COUNT_KERNEL_H_INCLUDED_

#include "hml_common.h"

/* TriangleCount kernel constants */
#define cHmlTriangleCountMaxNumKernels      32
#define cHmlTriangleCountThreadsPerBlock    512

#define cHmlTriangleCountSumThreadsPerBlock 256
#define cHmlTriangleCountSumBlocks          1024

/* if the number of successors of a vertex is in [minDeg, maxDeg),
 * then the kernel uses numThreads to expand the same vertex
 */
typedef struct {
  uint32_t   id;                /* id of the kernel */
  uint32_t   minDeg;            /* inclusive */
  uint32_t   maxDeg;            /* exclusive */
  uint32_t   minVertexRank;     /* inclusive */
  uint32_t   maxVertexRank;     /* exclusive */
  dim3       grid;
  dim3       block;
} HmlTriangleCountKernelArg;

__global__ void
hmlTriangleCountSumKernel(uint64_t        *blockCountArr,   /* output count */
                          const uint32_t  *countArr,
                          const uint32_t   maxSrcVertex);

typedef void (*HmlTriangleCountKernel)(uint32_t        *countArr,
                                       const uint32_t  *R,
                                       const uint32_t  *E,
                                       const uint32_t   maxSrcVertex,
                                       const uint32_t  *vertexRank,
                                       const uint32_t   minVertexRank,
                                       const uint32_t   maxVertexRank);

void
hmlTriangleCountKernelArgSet(HmlTriangleCountKernelArg *kernelArg,
                             uint32_t                   numPartitions,
                             uint32_t                  *minOutDeg,
                             uint32_t                  *partitionPrefixSize);

void
hmlTriangleCountKernelArgPrint(HmlTriangleCountKernelArg *kernelArg,
                               uint32_t                   numPartitions);

void
hmlTriangleCountKernelSetup(HmlTriangleCountKernel    *kernel,
                            uint32_t                  *minOutDegArr,
                            uint32_t                   maxNumKernels,
                            uint32_t                  *numPartitions,
                            HmlTriangleCountKernelArg *kernelArgArr);

#endif /* HML_TRIANGLE_COUNT_KERNEL_H_INCLUDED_ */
