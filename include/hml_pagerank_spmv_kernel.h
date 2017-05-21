/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_PAGERANK_SPMV_KERNEL_H_INCLUDED_
#define HML_PAGERANK_SPMV_KERNEL_H_INCLUDED_

#include "hml_common.h"

/* PagerankSpmv kernel constants */
#define cHmlPagerankSpmvMaxNumKernels      32
#define cHmlPagerankSpmvThreadsPerBlock    512

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
} HmlPagerankSpmvKernelArg;

typedef void (*PagerankSpmvKernel)(float           *outputMap,
                                   const uint32_t  *R,
                                   const uint32_t  *E,
                                   const uint32_t  *vertexRank,
                                   const uint32_t   minVertexRank,
                                   const uint32_t   maxVertexRank,
                                   const float      dampingFactor);

void
hmlPagerankSpmvKernelArgSet(HmlPagerankSpmvKernelArg *kernelArg,
                            uint32_t                  numPartitions,
                            uint32_t                 *minOutDeg,
                            uint32_t                 *vertexRank,
                            uint32_t                 *R,
                            uint32_t                 *partitionPrefixSize);

void
hmlPagerankSpmvKernelArgPrint(HmlPagerankSpmvKernelArg *kernelArg,
                              uint32_t numPartitions);

__global__ void
hmlPagerankSpmvInitKernel(float *devMap, float initVal, uint32_t numVertices);

void
hmlPagerankSpmvKernelSetup(PagerankSpmvKernel       *evenIterKernel,
                           PagerankSpmvKernel       *oddIterKernel,
                           uint32_t                 *minOutDegArr,
                           uint32_t                  maxNumKernels,
                           bool                      useTextureMem,
                           uint32_t                 *numPartitions,
                           HmlPagerankSpmvKernelArg *kernelArgArr);

#endif /* HML_PAGERANK_SPMV_KERNEL_H_INCLUDED_ */
