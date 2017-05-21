/*****************************************************************
*  Copyright (c) 2017. Palo Alto Research Center                *
*  All rights reserved.                                         *
*****************************************************************/

/*
 * hml_vertex_partition.c
 */
#include "hml_vertex_partition.h"

#define cHmlVertexPartitionEntriesInitSize        128

static HmlErrCode
hmlVertexPartitionEntriesInit(HmlVertexPartitionEntries *entries) {
  HML_ERR_PROLOGUE;

  memset(entries, 0, sizeof(*entries));
  MALLOC(entries->data, HmlVertexPartitionEntry,
         cHmlVertexPartitionEntriesInitSize);
  entries->size = cHmlVertexPartitionEntriesInitSize;

  HML_NORMAL_RETURN;
}

static bool
hmlVertexPartitionEntriesIsInitialized(HmlVertexPartitionEntries *entries) {
  if(!entries) {
    return false;
  }
  if(!entries->size) {
    return false;
  }
  if(entries->used) {
    return false;
  }
  return true;
}

static HmlErrCode
hmlVertexPartitionEntriesDelete(HmlVertexPartitionEntries *entries) {
  HML_ERR_PROLOGUE;

  FREE(entries->data);
  memset(entries, 0, sizeof(*entries));

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlVertexPartitionEntriesGrow(HmlVertexPartitionEntries *entries) {
  HML_ERR_PROLOGUE;

  REALLOC(entries->data, HmlVertexPartitionEntry, entries->size * 2);
  entries->size *= 2;

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlVertexPartitionEntriesAppend(HmlVertexPartitionEntries *entries,
                                uint32_t maxVertex, uint64_t numEdges) {
  HML_ERR_PROLOGUE;

  if(entries->used == entries->size) {
    HML_ERR_PASS(hmlVertexPartitionEntriesGrow(entries));
  }
  HML_ERR_GEN(entries->used >= entries->size, cHmlErrGeneral);
  entries->data[entries->used].maxVertex = maxVertex;
  entries->data[entries->used].numEdges = numEdges;
  entries->used++;

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlVertexPartitionEntriesGet(HmlVertexPartitionEntries *entries,
                             uint32_t idx, uint32_t *maxVertex, uint64_t *numEdges) {
  HML_ERR_PROLOGUE;

  HML_ERR_GEN(idx >= entries->used, cHmlErrGeneral);
  if(maxVertex) {
    *maxVertex = entries->data[idx].maxVertex;
  }
  if(numEdges) {
    *numEdges = entries->data[idx].numEdges;
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlVertexPartitionInit(HmlVertexPartition *vertexPartition) {
  HML_ERR_PROLOGUE;

  HML_ERR_GEN(!vertexPartition, cHmlErrGeneral);
  memset(vertexPartition, 0, sizeof(*vertexPartition));
  HML_ERR_PASS(hmlVertexPartitionEntriesInit(&vertexPartition->entries));

  HML_NORMAL_RETURN;
}

static bool
hmlVertexPartitionIsInitialized(HmlVertexPartition *vertexPartition) {
  if(!vertexPartition) {
    return false;
  }
  if(vertexPartition->numPartitions) {
    return false;
  }
  if(!hmlVertexPartitionEntriesIsInitialized(&vertexPartition->entries)) {
    return false;
  }
  return true;
}

HmlErrCode
hmlVertexPartitionDelete(HmlVertexPartition *vertexPartition) {
  HML_ERR_PROLOGUE;

  HML_ERR_GEN(!vertexPartition, cHmlErrGeneral);
  HML_ERR_PASS(hmlVertexPartitionEntriesDelete(&vertexPartition->entries));
  memset(vertexPartition, 0, sizeof(*vertexPartition));

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlVertexPartitionAddEntry(HmlVertexPartition *vertexPartition,
                           uint32_t              maxVertex,
                           uint64_t              numEdges) {
  HML_ERR_PROLOGUE;
  HML_ERR_PASS(hmlVertexPartitionEntriesAppend(&vertexPartition->entries,
               maxVertex, numEdges));
  vertexPartition->numPartitions++;
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlVertexPartitionGetEntry(HmlVertexPartition *vertexPartition,
                           uint32_t partition, uint32_t *maxVertex,
                           uint64_t *numEdges) {
  HML_ERR_PROLOGUE;
  HML_ERR_GEN(partition >= vertexPartition->numPartitions, cHmlErrGeneral);
  HML_ERR_PASS(hmlVertexPartitionEntriesGet(&vertexPartition->entries,
               partition, maxVertex, numEdges));
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlVertexPartitionGetMinMaxVertex(HmlVertexPartition *vertexPartition,
                                  uint32_t partition, uint32_t *minVertex,
                                  uint32_t *maxVertex) {
  HML_ERR_PROLOGUE;
  HML_ERR_GEN(partition >= vertexPartition->numPartitions, cHmlErrGeneral);
  if(partition > 0) {
    HML_ERR_PASS(hmlVertexPartitionEntriesGet(&vertexPartition->entries,
                 partition - 1, minVertex, NULL));
    ++(*minVertex);
  }
  else {
    *minVertex = 0;
  }
  HML_ERR_PASS(hmlVertexPartitionEntriesGet(&vertexPartition->entries,
               partition, maxVertex, NULL));

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlVertexPartitionComputeSize(uint32_t             *vertexDegreeArr,
                              uint32_t              numVertices,
                              size_t              maxNumEdgesPerPartition,
                              HmlVertexPartition *vertexPartition) {
  HML_ERR_PROLOGUE;
  uint32_t vertex;
  size_t numEdges;

  HML_ERR_GEN(!hmlVertexPartitionIsInitialized(vertexPartition), cHmlErrGeneral);
  numEdges = 0;
  for(vertex = 0; vertex < numVertices; ++vertex) {
    if(numEdges + vertexDegreeArr[vertex] <= maxNumEdgesPerPartition) {
      numEdges += vertexDegreeArr[vertex];
    }
    else {
      HML_ERR_PASS(hmlVertexPartitionAddEntry(vertexPartition, vertex - 1,
                                              numEdges));
      numEdges = vertexDegreeArr[vertex];
      HML_ERR_GEN(numEdges > maxNumEdgesPerPartition, cHmlErrGeneral);
    }
  }
  /* set the partition size for the last partition */
  HML_ERR_PASS(hmlVertexPartitionAddEntry(vertexPartition, numVertices - 1,
                                          numEdges));

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlVertexPartitionPrint(HmlVertexPartition *vertexPartition, FILE *file) {
  HML_ERR_PROLOGUE;
  uint32_t part;
  uint32_t prevMaxVertex = (uint32_t)-1;

  for(part = 0; part < vertexPartition->numPartitions; ++part) {
    uint32_t maxVertex;
    uint64_t numEdges;
    HML_ERR_PASS(hmlVertexPartitionGetEntry(vertexPartition, part, &maxVertex,
                                            &numEdges));
    fprintf(file, "; Info: partition#%d -> V = V[%12u - %12u], |E| = %12lu\n",
            part, (prevMaxVertex == (uint32_t)-1) ? 0 : prevMaxVertex + 1,
            maxVertex, numEdges);
    prevMaxVertex = maxVertex;
  }

  HML_NORMAL_RETURN;
}
