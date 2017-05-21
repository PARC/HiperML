/*****************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.               *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_VERTEX_PARTITION_H_INCLUDED_
#define HML_VERTEX_PARTITION_H_INCLUDED_

/* this file contains the prototypes for the routines that are used to
 * perform vertex partition operations.
 */

#include "hml_common.h"

/* make prototypes usable from C++ */
#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

typedef struct {
  uint32_t         maxVertex;
  uint64_t         numEdges;
} HmlVertexPartitionEntry;

typedef struct {
  HmlVertexPartitionEntry *data;
  uint32_t                   used;
  uint32_t                   size;
} HmlVertexPartitionEntries;

typedef struct {
  bool           srcVertexPartition;
  uint32_t         numPartitions;
  HmlVertexPartitionEntries entries;
  uint32_t         totalNumSrcVertices;
  uint32_t         totalNumDestVertices;
  size_t         totalNumEdges;
} HmlVertexPartition;

HmlErrCode
hmlVertexPartitionInit(HmlVertexPartition *vertexPartition);

HmlErrCode
hmlVertexPartitionDelete(HmlVertexPartition *vertexPartition);

HmlErrCode
hmlVertexPartitionComputeSize(uint32_t             *vertexDegreeArr,
                              uint32_t              numVertices,
                              size_t              maxNumEdgesPerPartition,
                              HmlVertexPartition *vertexPartition);

HmlErrCode
hmlVertexPartitionPrint(HmlVertexPartition *vertexPartition,
                        FILE               *file);

HmlErrCode
hmlVertexPartitionAddEntry(HmlVertexPartition *vertexPartition,
                           uint32_t              maxVertex,
                           uint64_t              numEdges);

HmlErrCode
hmlVertexPartitionGetEntry(HmlVertexPartition *vertexPartition,
                           uint32_t partition, uint32_t *maxVertex,
                           uint64_t *numEdges);

HmlErrCode
hmlVertexPartitionGetMinMaxVertex(HmlVertexPartition *vertexPartition,
                                  uint32_t partition, uint32_t *minVertex,
                                  uint32_t *maxVertex);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif  /* HML_VERTEX_PARTITION_H_INCLUDED_ */
