/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_GRAPH_CORE_H_INCLUDED_
#define HML_GRAPH_CORE_H_INCLUDED_

#include "hml_common.h"
#include "hml_vertex_partition.h"

#define cHmlGraphCoreInvalidVertex   ((uint32_t)-1)

typedef struct {
  uint32_t     *R;
  uint64_t      sizeofR;
  uint32_t     *E;
  uint64_t      sizeofE;
  uint64_t      numEdges;
  uint32_t      minSrcVertex;
  uint32_t      maxSrcVertex;
  uint32_t      minDestVertex;
  uint32_t      maxDestVertex;
} HmlGraph;

typedef struct {
  /* upper bound on numSrcVertices = maxSrcVertex - minSrcVertex + 1 */
  uint32_t      maxNumSrcVertices;
  /* lower bound on minSrcVertex below */
  uint32_t      minMinSrcVertex;
  /* degree of source vertices, unshifted for memory de-allocation,
   * |D0| = maxNumSrcVertices
   */
  uint32_t     *D0;
  /* shifted D0, s.t. &D[minMinSrcVertex] == &D0[0] */
  uint32_t     *D;
  /* R index array, unshifted for memory de-allocation,
   * |R0| = maxNumSrcVertices + 1
   */
  uint32_t     *R0;
  /* shifted R0, s.t. &R[minMinSrcVertex] == &R0[0] */
  uint32_t     *R;
  uint32_t     *E;
  uint32_t      numEdges;  /* |E| = numEdges */
  uint32_t      minSrcVertex;
  uint32_t      maxSrcVertex;
  uint32_t      minDestVertex;
  uint32_t      maxDestVertex;
  uint32_t      minOutDegree;  /* min. out-degree of vertices with successor(s) */
  uint32_t      maxOutDegree;
} HmlGraphCore;

typedef struct {
  uint32_t      numEdges;  /* |E| = numEdges */
  uint32_t      minSrcVertex;
  uint32_t      maxSrcVertex;
  uint32_t      minDestVertex;
  uint32_t      maxDestVertex;
  /* uint32_t     *R0; */
  /* uint32_t     *E;  */
} HmlGraphCoreFileHeader;

typedef struct {
  uint32_t prevSrcVertex;
  uint64_t numEdgesSoFar;
} HmlGraphCoreAppendState;

typedef enum {
  eAppendRegularEdge,
  eAppendEndofGraphEdge
} HmlGraphAppendEdgeCmd;

typedef struct {
  HmlGraphAppendEdgeCmd  cmd;
  uint32_t        *R;
  uint32_t        *E;
  uint64_t         numEdges;
  uint64_t         sizeofR;
  uint64_t         allocSizeofR;
  uint64_t         sizeofE;
  uint64_t         allocSizeofE;
  uint32_t        *r;
  uint32_t        *endofR;
  uint32_t        *e;
  uint32_t        *endofE;
  uint32_t         prevSrcVertex;
  uint32_t         minSrcVertex;
  uint32_t         maxSrcVertex;
  uint32_t         minDestVertex;
  uint32_t         maxDestVertex;
} HmlGraphAppendEdgeState;

typedef HmlErrCode(*HmlGraphCoreParaFunc)(HmlGraphCore *core,
    uint32_t        thread,
    uint32_t        minVertex,
    uint32_t        maxVertex,
    void         *args);

typedef struct {
  HmlGraphCore          *core;
  HmlVertexPartition    *partition;
  HmlGraphCoreParaFunc   func;
  void                  *args;
} HmlGraphCoreThreadArgs;

typedef HmlErrCode(*HmlGraphCoreFunc)(HmlGraphCore *core,
                                      void  *args,
                                      uint32_t srcVertex,
                                      uint32_t destVertex);

typedef HmlErrCode(*HmlGraphCoreConstFunc)(HmlGraphCore const *core,
    void  *args,
    uint32_t srcVertex,
    uint32_t destVertex);

typedef HmlErrCode(*HmlGraphCoreAppendIniter)(HmlGraphCore *core,
    void         *appendState);

typedef HmlErrCode(*HmlGraphCoreAppender)(HmlGraphCore *core,
    void         *appendState,
    uint32_t        srcVertex,
    uint32_t        destVertex);

typedef HmlErrCode(*HmlGraphCoreAppendFinalizer)(HmlGraphCore *core,
    void         *appendState);

typedef HmlErrCode(*HmlGraphCoreInsertIniter)(HmlGraphCore *core,
    void         *insertState);

typedef HmlErrCode(*HmlGraphCoreInserter)(HmlGraphCore *core,
    void         *insertState,
    uint32_t        srcVertex,
    uint32_t        destVertex);

typedef HmlErrCode(*HmlGraphCoreInsertFinalizer)(HmlGraphCore *core,
    void         *insertState);

/* make prototypes usable from C++ */
#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

void
hmlGraphAppendEdgeMemInit(HmlGraphAppendEdgeState *appState);

void hmlGraphAppendEdgeInit(HmlGraph *graph, HmlGraphAppendEdgeState *appState);

void
hmlGraphAppendEdge(uint32_t           srcVertex,
                   uint32_t           edgeId,
                   uint32_t           destVertex,
                   HmlGraphAppendEdgeState *appState);

void hmlGraphAppendEdgeFinal(HmlGraph *graph, HmlGraphAppendEdgeState *appState);

HmlErrCode
hmlGraphCoreAppendFromTsv2File(HmlGraphCore               *core,
                               FILE                       *file,
                               bool                        srcVertexOnRightColumn,
                               HmlGraphCoreAppendIniter    initer,
                               HmlGraphCoreAppender        appender,
                               HmlGraphCoreAppendFinalizer finalizer,
                               void                       *appendState);

void
hmlGraphReadTsv4(char *filename, bool sortedBySubject, HmlGraph *graph);

void
hmlGraphPrintEdges(FILE *file, HmlGraph *graph, bool sortedBySubject);

void
hmlGraphPrintStats(FILE *file, HmlGraph *graph);

HmlErrCode
hmlGraphCoreAppendInit(HmlGraphCore            *core,
                       HmlGraphCoreAppendState *state);

HmlErrCode
hmlGraphCoreAppend(HmlGraphCore            *core,
                   HmlGraphCoreAppendState *state,
                   uint32_t                 srcVertex,
                   uint32_t                 destVertex);

HmlErrCode
hmlGraphCoreAppendFinal(HmlGraphCore            *core,
                        HmlGraphCoreAppendState *state);

HmlErrCode
hmlGraphCoreVertexPartition(HmlGraphCore const *core,
                            uint64_t            maxNumEdgesPerPartition,
                            HmlVertexPartition *vertexPartition);

HmlErrCode
hmlGraphCoreCopyToGpu(HmlGraphCore *core, HmlGraphCore *gpuCore);

HmlErrCode
hmlGraphCoreGpuDelete(HmlGraphCore *core);

HmlErrCode
hmlGraphCoreInit(HmlGraphCore *core,
                 uint32_t        maxNumSrcVertices,
                 uint32_t        numEdges);

HmlErrCode
hmlGraphCoreDelete(HmlGraphCore *core);

HmlErrCode
hmlGraphCoreDeleteD(HmlGraphCore *core);

HmlErrCode
hmlGraphCoreSetR(HmlGraphCore *core,
                 const uint32_t *D,
                 uint32_t        minMinSrcVertex,
                 uint32_t        maxMaxSrcVertex);

HmlErrCode
hmlGraphCoreAddEdgeInit(HmlGraphCore *core);

HmlErrCode
hmlGraphCoreAddEdge(HmlGraphCore *core,
                    uint32_t        srcVertex,
                    uint32_t        destVertex);

void
hmlGraphCorePrintStats(HmlGraphCore *core, FILE *file);

HmlErrCode
hmlGraphCoreCountBidirectDegree(HmlGraphCore const *core,
                                uint32_t             *D,
                                uint32_t              numVertices);

HmlErrCode
hmlGraphCoreCountDegreeIfSmallerP(HmlGraphCore const *core,
                                  uint32_t const       *P,
                                  uint32_t             *D,
                                  uint32_t              numVertices);

HmlErrCode
hmlGraphCoreDefaultInsertIniter(HmlGraphCore *core, void *insertState);

HmlErrCode
hmlGraphCoreDefaultInserter(HmlGraphCore *core,
                            void         *insertState,
                            uint32_t        srcVertex,
                            uint32_t        destVertex);

HmlErrCode
hmlGraphCoreInsertFromSrc(HmlGraphCore               *core,
                          HmlGraphCore const         *src,
                          HmlGraphCoreInsertIniter    initer,
                          HmlGraphCoreInserter        inserter,
                          HmlGraphCoreInsertFinalizer finalizer,
                          void                       *insertState);

HmlErrCode
hmlGraphCoreSortEdges(HmlGraphCore *core, uint32_t numThreads);

HmlErrCode
hmlGraphCoreSortEdgesByPartition(HmlGraphCore       *core,
                                 HmlVertexPartition *partition);

HmlErrCode
hmlGraphCoreRunParaFuncByPartition(HmlGraphCore          *core,
                                   HmlGraphCoreParaFunc   func,
                                   void                  *args,
                                   HmlVertexPartition    *partition);

/*! Reads a two-column tab-separated graph core from file \a file into
 * an HmlGraphCore struct 'core'
 */
HmlErrCode
hmlGraphCoreReadTsv2File(HmlGraphCore *core,
                         FILE         *file,
                         bool          srcVertexOnRightColumn);

HmlErrCode
hmlGraphCoreRangeReadTsv2File(HmlGraphCore *core,
                              FILE         *file,
                              bool          srcVertexOnRightColumn,
                              uint32_t        minMinSrcVertex,
                              uint32_t        maxMaxSrcVertex);

HmlErrCode
hmlGraphCoreReadTsv2FileWithName(HmlGraphCore *core,
                                 const char   *fileName,
                                 bool          srcVertexOnRightColumn);

HmlErrCode
hmlGraphCoreWriteBinaryFile(const HmlGraphCore *core,
                            FILE               *file);

HmlErrCode
hmlGraphCoreReadBinaryFile(HmlGraphCore *core,
                           FILE         *file);

HmlErrCode
hmlGraphCoreWriteTsv2File(const HmlGraphCore *core,
                          FILE               *file,
                          bool                srcVertexOnRightColumn);

HmlErrCode
hmlGraphCoreWriteVertexInfoTsv3File(const HmlGraphCore *core,
                                    FILE               *file,
                                    bool                srcVertexOnRightColumn,
                                    uint32_t             *vertexInfoArr,
                                    bool                destVertexInfo);

HmlErrCode
hmlGraphCoreWriteVertexInfoTsv4File(const HmlGraphCore *core,
                                    FILE               *file,
                                    bool                srcVertexOnRightColumn,
                                    uint16_t              edgeType,
                                    uint32_t             *vertexInfoArr,
                                    bool                destVertexInfo);

HmlErrCode
hmlGraphCoreWriteEdgeInfoTsv3File(const HmlGraphCore *core,
                                  FILE               *file,
                                  bool                srcVertexOnRightColumn,
                                  uint32_t             *edgeInfoArr);
HmlErrCode
hmlGraphCoreWriteEdgeInfoTsv4File(const HmlGraphCore *core,
                                  FILE               *file,
                                  bool                srcVertexOnRightColumn,
                                  uint16_t              edgeType,
                                  uint32_t             *edgeInfoArr);

HmlErrCode
hmlGraphCoreBindTexture(HmlGraphCore            *core,
                        const texture<uint32_t, 1> &texDataR,
                        const texture<uint32_t, 1> &texDataE);

HmlErrCode
hmlGraphCoreUnbindTexture(const texture<uint32_t, 1> &texDataR,
                          const texture<uint32_t, 1> &texDataE);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* HML_GRAPH_CORE_H_INCLUDED_ */
