/*****************************************************************
*  Copyright (c) 2014. Palo Alto Research Center                *
*  All rights reserved.                                         *
*****************************************************************/

/* hml_graph_core.c
 */
#include "hml_graph_core.h"
#include "hml_thread_group.h"
#include "hml_file_utils.h"

#define cHmlTsv2LineSizeMax           1024
#define cHmlSuccIndexArrayInitSize    100000
#define cHmlSuccArrayInitSize         400000
#define cHmlGraphCoreRSizeInit        1024
#define cHmlGraphCoreESizeInit        1024

void
hmlGraphAppendEdgeMemInit(HmlGraphAppendEdgeState *appState) {
  MALLOC(appState->E, uint32_t, appState->allocSizeofE);
  appState->e = appState->E;
  appState->endofE = &appState->E[appState->allocSizeofE];
  CALLOC(appState->R, uint32_t, appState->allocSizeofR);
  appState->r = appState->R;
  appState->endofR = &appState->R[appState->allocSizeofR];
}

void hmlGraphAppendEdgeInit(HmlGraph *graph, HmlGraphAppendEdgeState *appState) {
  appState->cmd = eAppendRegularEdge;
  appState->numEdges = 0;
  appState->sizeofR = 0;
  appState->allocSizeofR = cHmlSuccIndexArrayInitSize;
  appState->sizeofE = 0;
  appState->allocSizeofE = cHmlSuccArrayInitSize;
  appState->r = NULL;
  appState->e = NULL;
  appState->prevSrcVertex = (uint32_t)-1;
  appState->minSrcVertex = (uint32_t)-1;
  appState->maxSrcVertex = 0;
  appState->minDestVertex = (uint32_t)-1;
  appState->maxDestVertex = 0;
  hmlGraphAppendEdgeMemInit(appState);
}

void
hmlGraphAppendEdge(uint32_t           srcVertex,
                   uint32_t           edgeId,
                   uint32_t           destVertex,
                   HmlGraphAppendEdgeState *appState) {
  appState->minDestVertex = min(appState->minDestVertex, destVertex);
  appState->maxDestVertex = max(appState->maxDestVertex, destVertex);
  /* make sure there is enough room in E.
   * note that each entry consumes two 32-bit words:
   *   1st word: a 32-bit unsigned integer to store 'destVertex'
   *   2nd word: a 32-bit unsigned integer to store 'edgeId'
   */
  if(appState->e + 1 >= appState->endofE) {
    appState->allocSizeofE *= 2;
    REALLOC(appState->E, uint32_t, appState->allocSizeofE);
    appState->endofE = &appState->E[appState->allocSizeofE];
    appState->e = &appState->E[appState->sizeofE];
  }
  if(srcVertex != appState->prevSrcVertex) {
    appState->minSrcVertex = min(appState->minSrcVertex, srcVertex);
    appState->maxSrcVertex = max(appState->maxSrcVertex, srcVertex);
    ++appState->prevSrcVertex;
    /* need extra slot for the dummy last element of R */
    if(appState->r + (srcVertex - appState->prevSrcVertex + 1) >=
        appState->endofR) {
      appState->allocSizeofR *= 2;
      appState->allocSizeofR =
        max(appState->allocSizeofR,
            (uint64_t)(appState->r + (srcVertex - appState->prevSrcVertex + 2)
                       - appState->R));
      REALLOC(appState->R, uint32_t, appState->allocSizeofR);
      appState->endofR = &appState->R[appState->allocSizeofR];
      appState->r = &appState->R[appState->sizeofR];
    }
    if(srcVertex > appState->prevSrcVertex) {
      for(; appState->prevSrcVertex < srcVertex; ++appState->prevSrcVertex) {
        /* fill in dummy entry for each missing srcVertex */
        *appState->r++ = (uint32_t)(appState->e - appState->E);
        ++appState->sizeofR;
      }
      /* we don't store the dummy vertex -1
       * *appState->e++ = (uint32_t)-1;
       * ++appState->sizeofE;
       */
    }
    *appState->r++ = (uint32_t)(appState->e - appState->E);
    ++appState->sizeofR;
  }
  *appState->e++ = destVertex;
  *appState->e++ = edgeId;
  appState->sizeofE += 2;
  ++appState->numEdges;
}

void hmlGraphAppendEdgeFinal(HmlGraph *graph, HmlGraphAppendEdgeState *appState) {
  *appState->r = (uint32_t)(appState->e - appState->E);
  ++appState->sizeofR;
  REALLOC(appState->R, uint32_t, appState->sizeofR);
  REALLOC(appState->E, uint32_t, appState->sizeofE);
  graph->R = appState->R;
  graph->sizeofR = appState->sizeofR;
  graph->E = appState->E;
  graph->sizeofE = appState->sizeofE;
  graph->numEdges = appState->numEdges;
  graph->minSrcVertex = appState->minSrcVertex;
  graph->maxSrcVertex = appState->maxSrcVertex;
  graph->minDestVertex = appState->minDestVertex;
  graph->maxDestVertex = appState->maxDestVertex;
}

void
hmlGraphReadTsv4(char *filename, bool sortedBySubject, HmlGraph *graph) {
  FILE                         *file;
  char                          line[cHmlLineBufferSize];
  char                         *str;
  uint32_t                        charInt;
  uint32_t                        edgeId;
  uint32_t                        srcVertex;
  uint32_t                        destVertex;
  uint32_t                        edgeType;
  HmlGraphAppendEdgeState            appStateVal;
  HmlGraphAppendEdgeState           *appState = &appStateVal;

  file = fopen(filename, "rb");
  if(!file) {
    fprintf(stderr, "; Error: Cannot open file: %s\n", filename);
    exit(EXIT_FAILURE);
  }
  hmlGraphAppendEdgeInit(graph, appState);
  for(;;) {
    /* Go get the next <srcVertex, edgeType, edgeId, destVertex> quadruple
     * if it exists.
     */
    str = fgets(line, cHmlLineBufferSize, file);
    if(!str) {
      break;
    }
    /* are quadruples sorted by subject? */
    if(sortedBySubject) {
      /* the four while loops below achieve the same
       * function as:
       * fscanf(file, "%u %u %u %u\n",
       *   &srcVertex, &edgeType, &edgeId, &destVertex);
       */
      srcVertex = 0;
      while((charInt = (uint32_t)*str++) != ' ') {
        srcVertex = srcVertex * 10 + charInt - '0';
      }
      edgeType = 0;
      while((charInt = (uint32_t)*str++) != ' ') {
        edgeType = edgeType * 10 + charInt - '0';
      }
      edgeId = 0;
      while((charInt = (uint32_t)*str++) != ' ') {
        edgeId = edgeId * 10 + charInt - '0';
      }
      /* we could compute the inverse of number of successors:
       *   oneOverEdgeId = (float) 1.0 / (float) edgeId;
       * and then pretend edgeId is a uint32_t, as follows:
       *   *((float*)&edgeId) = oneOverEdgeId;
       * but currently we don't do these tricks, because
       * 1. it doesn't improve speed
       * 2. it complicates the code
       */
      destVertex = 0;
      while((charInt = (uint32_t)*str++) != '\n') {
        destVertex = destVertex * 10 + charInt - '0';
      }
    }
    else { /* if not by subject, they must be sorted by object */
      /* the four while loops below achieve the same
       * function as:
       * fscanf(file, "%u %u %u %u\n",
       *   &destVertex, &edgeType, &edgeId, &srcVertex);
       */
      destVertex = 0;
      while((charInt = (uint32_t)*str++) != ' ') {
        destVertex = destVertex * 10 + charInt - '0';
      }
      edgeType = 0;
      while((charInt = (uint32_t)*str++) != ' ') {
        edgeType = edgeType * 10 + charInt - '0';
      }
      edgeId = 0;
      while((charInt = (uint32_t)*str++) != ' ') {
        edgeId = edgeId * 10 + charInt - '0';
      }
      /* we could compute the inverse of number of successors:
       *   oneOverEdgeId = (float) 1.0 / (float) edgeId;
       * and then pretend edgeId is a uint32_t, as follows:
       *   *((float*)&edgeId) = oneOverEdgeId;
       * but currently we don't do these tricks, because:
       * 1. it doesn't improve speed
       * 2. it complicates the code
       */
      srcVertex = 0;
      while((charInt = (uint32_t)*str++) != '\n') {
        srcVertex = srcVertex * 10 + charInt - '0';
      }
    }
    hmlGraphAppendEdge(srcVertex, edgeId, destVertex, appState);
    if(feof(file)) {
      break;
    }
  }
  hmlGraphAppendEdgeFinal(graph, appState);
}

void
hmlGraphPrintEdges(FILE *file, HmlGraph *graph, bool sortedBySubject) {
  uint32_t  vid;
  uint32_t *succ;
  uint32_t *succMax;
  uint32_t *R = graph->R;
  uint32_t *E = graph->E;

  for(vid = graph->minSrcVertex; vid <= graph->maxSrcVertex; ++vid) {
    succ = &E[R[vid]];
    succMax = &E[R[vid + 1]];
    while(succ < succMax) {
      if(sortedBySubject) {
        fprintf(file, "%d 0 %d %d\n", vid, succ[1], succ[0]);
      }
      else {
        fprintf(file, "%d 0 %d %d\n", succ[0], succ[1], vid);
      }
      succ += 2;
    }
  }
}

void
hmlGraphPrintStats(FILE *file, HmlGraph *graph) {
  size_t byteSizeofR = graph->sizeofR * sizeof(uint32_t);
  size_t byteSizeofE = graph->sizeofE * sizeof(uint32_t);
  size_t byteSizeofGraph = byteSizeofR + byteSizeofE;

  fprintf(file, "min_source_vertex_id=%d\n", graph->minSrcVertex);
  fprintf(file, "max_source_vertex_id=%d\n", graph->maxSrcVertex);
  fprintf(file, "min_destination_vertex_id=%d\n", graph->minDestVertex);
  fprintf(file, "max_destination_vertex_id=%d\n", graph->maxDestVertex);
  fprintf(file, "num_edges=%ld\n", graph->numEdges);
  fprintf(file, "bytes_of_edge_list=%ld\n", byteSizeofR);
  fprintf(file, "bytes_of_edge_index=%ld\n", byteSizeofE);
  fprintf(file, "; total graph size = %.2lf (MB)\n",
          (double)byteSizeofGraph / (1024 * 1024));
}

HmlErrCode
hmlGraphCoreCopyToGpu(HmlGraphCore *core, HmlGraphCore *gpuCore) {
  HML_ERR_PROLOGUE;

  /* since we don't copy the D[] array, better make sure it is null */
  HML_ERR_GEN(core->D || core->D0, cHmlErrGeneral);

  /* make a shallow copy of core in gpuCore */
  memcpy(gpuCore, core, sizeof(HmlGraphCore));

  /* allocate R0[] on device, + 1 for the dummy R[v] */
  HANDLE_ERROR(cudaMalloc(&gpuCore->R0,
                          sizeof(uint64_t) * (core->maxNumSrcVertices + 1)));
  gpuCore->R = gpuCore->R0 - gpuCore->minMinSrcVertex;

  /* allocate E[] on device */
  HANDLE_ERROR(cudaMalloc(&gpuCore->E,
                          sizeof(uint32_t) * core->numEdges));

  /* copy host R0[] to device R0[], + 1 for the dummy R[v] */
  HANDLE_ERROR(cudaMemcpy(gpuCore->R0, core->R0,
                          sizeof(uint64_t) * (core->maxNumSrcVertices + 1),
                          cudaMemcpyHostToDevice));

  /* copy host E[] to device E[] */
  HANDLE_ERROR(cudaMemcpy(gpuCore->E, core->E,
                          sizeof(uint32_t) * core->numEdges,
                          cudaMemcpyHostToDevice));
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreGpuDelete(HmlGraphCore *core) {
  HML_ERR_PROLOGUE;
  HML_ERR_GEN(core->D && core->D + core->minMinSrcVertex != core->D0,
              cHmlErrGeneral);
  HANDLE_ERROR(cudaFree(core->D0));
  HML_ERR_GEN(core->R && core->R + core->minMinSrcVertex != core->R0,
              cHmlErrGeneral);
  HANDLE_ERROR(cudaFree(core->R0));
  HANDLE_ERROR(cudaFree(core->E));
  memset(core, 0, sizeof(HmlGraphCore));
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreInit(HmlGraphCore *core,
                 uint32_t        maxNumSrcVertices,
                 uint32_t        numEdges) {
  HML_ERR_PROLOGUE;

  memset((void *)core, 0, sizeof(HmlGraphCore));
  CALLOC(core->D0, uint32_t, maxNumSrcVertices);
  CALLOC(core->R0, uint32_t, maxNumSrcVertices + 1);
  MALLOC(core->E, uint32_t, numEdges);
  core->maxNumSrcVertices = maxNumSrcVertices;
  core->numEdges          = numEdges;
  core->minSrcVertex      = (uint32_t)-1;
  core->minDestVertex     = (uint32_t)-1;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreDelete(HmlGraphCore *core) {
  HML_ERR_PROLOGUE;

  HML_ERR_GEN(core->D && core->D + core->minMinSrcVertex != core->D0,
              cHmlErrGeneral);
  FREE(core->D0);
  HML_ERR_GEN(core->R && core->R + core->minMinSrcVertex != core->R0,
              cHmlErrGeneral);
  FREE(core->R0);
  FREE(core->E);
  memset((void *)core, 0, sizeof(core));

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreSetR(HmlGraphCore *core,
                 uint32_t const *D,
                 uint32_t        minMinSrcVertex,
                 uint32_t        maxMaxSrcVertex) {
  HML_ERR_PROLOGUE;
  uint32_t  v;
  uint32_t *R;
  uint32_t  maxNumSrcVertices = maxMaxSrcVertex - minMinSrcVertex + 1;
  uint32_t  numEdges = 0;

  HML_ERR_GEN(core->maxNumSrcVertices < maxNumSrcVertices, cHmlErrGeneral);
  HML_ERR_GEN(core->R, cHmlErrGeneral);
  /* shift D such that &D[minMinSrcVertex] == &D0[0] */
  core->D = core->D0 - minMinSrcVertex;
  /* shift R such that &R[minMinSrcVertex] == &R0[0] */
  core->R = core->R0 - minMinSrcVertex;
  R = core->R;
  for(v = minMinSrcVertex; v <= maxMaxSrcVertex; ++v) {
    R[v] = numEdges;
    numEdges += D[v];
  }
  R[v] = numEdges; /* yes, v = maxMaxSrcVertex + 1 */
  HML_ERR_GEN(core->numEdges != numEdges, cHmlErrGeneral);
  core->minMinSrcVertex = minMinSrcVertex;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreAddEdgeInit(HmlGraphCore *core) {
  HML_ERR_PROLOGUE;
  uint32_t maxMaxSrcVertex = core->minMinSrcVertex + core->maxNumSrcVertices - 1;

  /* make sure hmlGraphCoreSetR() has been called already */
  HML_ERR_GEN(!core->R, cHmlErrGeneral);
  /* core->D should be set to all zero's, which is expensive to verify,
  * so core->D[minMinSrcVertex] (i.e., core->D0[0]) is used as a cheap check,
  * which is better than no check at all
  */
  HML_ERR_GEN(!core->D || core->D[core->minMinSrcVertex], cHmlErrGeneral);
  /* core->R should have been already set before calling this function */
  HML_ERR_GEN(core->R[maxMaxSrcVertex + 1] != core->numEdges, cHmlErrGeneral);
  /* init min,maxSrcVertex stats */
  core->minSrcVertex = (uint32_t)-1;
  core->maxSrcVertex = 0;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreAddEdge(HmlGraphCore *core,
                    uint32_t      srcVertex,
                    uint32_t      destVertex) {
  HML_ERR_PROLOGUE;

  /* make sure there is enough space left before the next vertex */
  HML_ERR_GEN(core->R[srcVertex] + core->D[srcVertex] >= core->R[srcVertex + 1],
              cHmlErrGeneral);
  /* note the ++ operator below */
  core->E[core->R[srcVertex] + core->D[srcVertex]++] = destVertex;
  core->minSrcVertex = min(core->minSrcVertex, srcVertex);
  core->maxSrcVertex = max(core->maxSrcVertex, srcVertex);
  core->minDestVertex = min(core->minDestVertex, destVertex);
  core->maxDestVertex = max(core->maxDestVertex, destVertex);

  HML_NORMAL_RETURN;
}

void
hmlGraphCorePrintStats(HmlGraphCore *core, FILE *file) {
  size_t byteSizeofR = core->maxNumSrcVertices * sizeof(core->R[0]);
  size_t byteSizeofE = core->numEdges * sizeof(core->E[0]);
  size_t byteSizeofGraph = byteSizeofR + byteSizeofE;

  fprintf(file, "min_source_vertex_id=%d\n", core->minSrcVertex);
  fprintf(file, "max_source_vertex_id=%d\n", core->maxSrcVertex);
  fprintf(file, "min_destination_vertex_id=%d\n", core->minDestVertex);
  fprintf(file, "max_destination_vertex_id=%d\n", core->maxDestVertex);
  fprintf(file, "num_edges=%u\n", core->numEdges);
  fprintf(file, "bytes_of_edge_list=%ld\n", byteSizeofR);
  fprintf(file, "bytes_of_edge_index=%ld\n", byteSizeofE);
  fprintf(file, "; total core size = %.2lf (MB)\n",
          (double)byteSizeofGraph / (1024 * 1024));
}

HmlErrCode
hmlGraphCoreAppendInit(HmlGraphCore            *core,
                       HmlGraphCoreAppendState *state) {
  memset(core, 0, sizeof(HmlGraphCore));
  memset(state, 0, sizeof(HmlGraphCoreAppendState));
  state->prevSrcVertex = cHmlGraphCoreInvalidVertex;
  MALLOC(core->R0, uint32_t, cHmlGraphCoreRSizeInit);
  MALLOC(core->E, uint32_t, cHmlGraphCoreESizeInit);
  core->minSrcVertex = (uint32_t)-1;
  core->minDestVertex = (uint32_t)-1;
  core->minOutDegree = (uint32_t)-1;
  core->maxNumSrcVertices = cHmlGraphCoreRSizeInit;
  core->numEdges = cHmlGraphCoreESizeInit;
  return cHmlErrSuccess;
}

HmlErrCode
hmlGraphCoreAppend(HmlGraphCore            *core,
                   HmlGraphCoreAppendState *state,
                   uint32_t                 srcVertex,
                   uint32_t                 destVertex) {
  HML_ERR_PROLOGUE;
  uint32_t outDegree;

  HML_ERR_GEN(state->prevSrcVertex > srcVertex &&
              state->prevSrcVertex != cHmlGraphCoreInvalidVertex, cHmlErrGeneral);
  if(state->prevSrcVertex == cHmlGraphCoreInvalidVertex) {
    core->minMinSrcVertex = core->minSrcVertex = srcVertex;
    HML_ERR_GEN(core->R, cHmlErrGeneral);
    core->R = core->R0 - srcVertex;
    core->R[srcVertex] = 0;
    state->prevSrcVertex = srcVertex;
  }
  if(srcVertex - core->minMinSrcVertex >= core->maxNumSrcVertices) {
    uint32_t newRSize = max(core->maxNumSrcVertices * 2,
                            srcVertex - core->minMinSrcVertex + 1);
    REALLOC(core->R0, uint32_t, newRSize);
    core->maxNumSrcVertices = newRSize;
    core->R = core->R0 - core->minMinSrcVertex;
  }
  if(state->prevSrcVertex < srcVertex) {
    outDegree = (uint32_t)(state->numEdgesSoFar - core->R[state->prevSrcVertex]);
    core->minOutDegree = min(core->minOutDegree, outDegree);
    core->maxOutDegree = max(core->maxOutDegree, outDegree);
    while(state->prevSrcVertex < srcVertex) {
      core->R[++state->prevSrcVertex] = state->numEdgesSoFar;
    }
  }
  if(state->numEdgesSoFar >= core->numEdges) {
    REALLOC(core->E, uint32_t, core->numEdges * 2);
    core->numEdges *= 2;
  }
  core->E[state->numEdgesSoFar++] = destVertex;
  /* no need to update minSrcVertex and maxSrcVertex for ORDERED tsv2 file */
  /* core->minSrcVertex = min(core->minSrcVertex, srcVertex); */
  /* core->maxSrcVertex = max(core->maxSrcVertex, srcVertex); */
  core->minDestVertex = min(core->minDestVertex, destVertex);
  core->maxDestVertex = max(core->maxDestVertex, destVertex);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreAppendFinal(HmlGraphCore            *core,
                        HmlGraphCoreAppendState *state) {
  core->maxSrcVertex = state->prevSrcVertex;
  core->maxNumSrcVertices = core->maxSrcVertex - core->minMinSrcVertex + 1;
  REALLOC(core->R0, uint32_t, core->maxNumSrcVertices + 1);
  core->R = core->R0 - core->minMinSrcVertex;
  core->R[core->maxSrcVertex + 1] = state->numEdgesSoFar;
  if(state->numEdgesSoFar) {  /* make sure there is at least one edge */
    REALLOC(core->E, uint32_t, state->numEdgesSoFar);
  }
  else {
    FREE(core->E);
  }
  core->numEdges = state->numEdgesSoFar;

  return cHmlErrSuccess;
}

HmlErrCode
hmlGraphCoreAppendFromTsv2File(HmlGraphCore               *core,
                               FILE                       *file,
                               bool                        srcVertexOnRightColumn,
                               HmlGraphCoreAppendIniter    initer,
                               HmlGraphCoreAppender        appender,
                               HmlGraphCoreAppendFinalizer finalizer,
                               void                       *appendState) {
  HML_ERR_PROLOGUE;
  char    line[cHmlTsv2LineSizeMax];
  char   *str;
  uint32_t  digit;
  uint32_t  srcVertex;
  uint32_t  destVertex;
  uint32_t  tmpVertex;

  HML_ERR_PASS(initer(core, appendState));
  while(!feof(file)) {
    /* Go get the next <srcVertex, destVertex> tuple if it exists */
    str = fgets(line, cHmlTsv2LineSizeMax, file);
    if(!str) {
      break;
    }
    /* the two while loops below achieve the same
     * function as:
     * fscanf(file, "%u %u\n", &srcVertex, &destVertex);
     */
    srcVertex = 0;
    while((digit = (uint32_t)*str++) != ' ' && digit != '\t') {
      HML_ERR_GEN(digit < '0' || digit > '9', cHmlErrGeneral);
      srcVertex = srcVertex * 10 + digit - '0';
    }
    destVertex = 0;
    while((digit = (uint32_t)*str++) != '\n') {
      HML_ERR_GEN(digit < '0' || digit > '9', cHmlErrGeneral);
      destVertex = destVertex * 10 + digit - '0';
    }
    if(srcVertexOnRightColumn) {
      tmpVertex = srcVertex;
      srcVertex = destVertex;
      destVertex = tmpVertex;
    }
    HML_ERR_PASS(appender(core, appendState, srcVertex, destVertex));
  }
  HML_ERR_PASS(finalizer(core, appendState));

  HML_NORMAL_RETURN;
}

/*! Reads a two-column tab-separated graph core from file \a file into
 * an HmlGraphCore struct 'core'
 */
HmlErrCode
hmlGraphCoreReadTsv2File(HmlGraphCore *core,
                         FILE         *file,
                         bool          srcVertexOnRightColumn) {
  HML_ERR_PROLOGUE;
  uint32_t  numEdges = 0;
  char    line[cHmlTsv2LineSizeMax];
  char   *str;
  uint32_t  digit;
  uint32_t  srcVertex;
  uint32_t  minMinSrcVertex = core->minMinSrcVertex;
  uint32_t  maxMaxSrcVertex = minMinSrcVertex + core->maxNumSrcVertices - 1;
  uint32_t  destVertex;

  HML_ERR_PASS(hmlGraphCoreAddEdgeInit(core));
  for(;;) {
    /* Go get the next <srcVertex, destVertex> tuple if it exists */
    str = fgets(line, cHmlTsv2LineSizeMax, file);
    if(!str) {
      break;
    }
    /* the two while loops below achieve the same
     * function as:
     * fscanf(file, "%u %u\n", &srcVertex, &destVertex);
     */
    srcVertex = 0;
    while((digit = (uint32_t)*str++) != ' ' && digit != '\t') {
      srcVertex = srcVertex * 10 + digit - '0';
    }
    destVertex = 0;
    while((digit = (uint32_t)*str++) != '\n') {
      destVertex = destVertex * 10 + digit - '0';
    }

    if(!srcVertexOnRightColumn) {
      /* the actual min,max range can be narrower or the same, but not wider */
      HML_ERR_GEN(srcVertex < minMinSrcVertex || srcVertex > maxMaxSrcVertex,
                  cHmlErrGeneral);
      HML_ERR_PASS(hmlGraphCoreAddEdge(core, srcVertex, destVertex));
    }
    else {
      /* the actual min,max range can be narrower or the same, but not wider */
      HML_ERR_GEN(destVertex < minMinSrcVertex || destVertex > maxMaxSrcVertex,
                  cHmlErrGeneral);
      HML_ERR_PASS(hmlGraphCoreAddEdge(core, destVertex, srcVertex));
    }
    ++numEdges;
    HML_ERR_GEN(numEdges > core->numEdges, cHmlErrGeneral);
    if(feof(file)) {
      break;
    }
  }
  HML_ERR_GEN(core->numEdges != numEdges, cHmlErrGeneral);
  /* the following two checks are redundant but just in case */
  HML_ERR_GEN(core->minSrcVertex < minMinSrcVertex, cHmlErrGeneral);
  HML_ERR_GEN(core->maxSrcVertex > maxMaxSrcVertex, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreVertexPartition(HmlGraphCore const *core,
                            uint64_t              maxNumEdgesPerPartition,
                            HmlVertexPartition *vertexPartition) {
  HML_ERR_PROLOGUE;
  uint32_t vertex;
  uint64_t outDegree;
  uint64_t numEdges;

  HML_ERR_GEN(!vertexPartition, cHmlErrGeneral);
  HML_ERR_GEN(!vertexPartition->entries.size, cHmlErrGeneral);
  HML_ERR_GEN(vertexPartition->entries.used, cHmlErrGeneral);
  numEdges = 0;
  for(vertex = core->minSrcVertex; vertex <= core->maxSrcVertex; ++vertex) {
    outDegree = core->R[vertex + 1] - core->R[vertex];
    if(numEdges + outDegree <= maxNumEdgesPerPartition) {
      numEdges += outDegree;
    }
    else {
      HML_ERR_PASS(hmlVertexPartitionAddEntry(vertexPartition, vertex - 1,
                                              numEdges));
      numEdges = outDegree;
      HML_ERR_GEN(numEdges > maxNumEdgesPerPartition, cHmlErrGeneral);
    }
  }
  /* set the partition size for the last partition */
  HML_ERR_PASS(hmlVertexPartitionAddEntry(vertexPartition, core->maxSrcVertex,
                                          numEdges));

  HML_NORMAL_RETURN;
}

/*! Reads a two-column tab-separated graph core from file \a file into
 * an HmlGraphCore struct 'core' but only include edges whose
 * source vertex falls in the range of [minMinSrcVertex, maxMaxSrcVertex],
 * inclusively.
 * This function differs from hmlGraphCoreReadTsv2File in that it won't
 * complain if the source vertex falls out of the [min,max] range
 */
HmlErrCode
hmlGraphCoreRangeReadTsv2File(HmlGraphCore *core,
                              FILE         *file,
                              bool          srcVertexOnRightColumn,
                              uint32_t        minMinSrcVertex,
                              uint32_t        maxMaxSrcVertex) {
  HML_ERR_PROLOGUE;
  uint32_t   numEdges = 0;
  char     line[cHmlTsv2LineSizeMax];
  char    *str;
  uint32_t   digit;
  uint32_t   srcVertex;
  uint32_t   maxNumSrcVertices = maxMaxSrcVertex - minMinSrcVertex + 1;
  uint32_t   destVertex;

  HML_ERR_PASS(hmlGraphCoreAddEdgeInit(core));
  /* make sure the core has enough storage to accommodate maxNumSrcVertices */
  HML_ERR_GEN(core->maxNumSrcVertices < maxNumSrcVertices, cHmlErrGeneral);
  HML_ERR_GEN(core->minMinSrcVertex > minMinSrcVertex, cHmlErrGeneral);
  for(;;) {
    /* Go get the next <srcVertex, destVertex> tuple if it exists */
    str = fgets(line, cHmlTsv2LineSizeMax, file);
    if(!str) {
      break;
    }
    /* the two while loops below achieve the same
     * function as:
     * fscanf(file, "%u %u\n", &srcVertex, &destVertex);
     */
    srcVertex = 0;
    while((digit = (uint32_t)*str++) != ' ' && digit != '\t') {
      srcVertex = srcVertex * 10 + digit - '0';
    }
    destVertex = 0;
    while((digit = (uint32_t)*str++) != '\n') {
      destVertex = destVertex * 10 + digit - '0';
    }

    if(!srcVertexOnRightColumn) {
      /* the actual min,max range can be narrower or the same, but not wider */
      if(srcVertex >= minMinSrcVertex && srcVertex <= maxMaxSrcVertex) {
        HML_ERR_PASS(hmlGraphCoreAddEdge(core, srcVertex, destVertex));
        ++numEdges;
      }
    }
    else {
      /* the actual min,max range can be narrower or the same, but not wider */
      if(destVertex >= minMinSrcVertex && destVertex <= maxMaxSrcVertex) {
        HML_ERR_PASS(hmlGraphCoreAddEdge(core, destVertex, srcVertex));
        ++numEdges;
      }
    }
    HML_ERR_GEN(numEdges > core->numEdges, cHmlErrGeneral);
    if(feof(file)) {
      break;
    }
  }
  HML_ERR_GEN(core->numEdges != numEdges, cHmlErrGeneral);
  /* the following two checks are redundant but just in case */
  HML_ERR_GEN(core->minSrcVertex < minMinSrcVertex, cHmlErrGeneral);
  HML_ERR_GEN(core->maxSrcVertex > maxMaxSrcVertex, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreReadTsv2FileWithName(HmlGraphCore *core,
                                 const char   *fileName,
                                 bool          srcVertexOnRightColumn) {
  HML_ERR_PROLOGUE;
  FILE      *file;

  HML_ERR_PASS(hmlFileOpenRead(fileName, &file));
  HML_ERR_PASS(hmlGraphCoreReadTsv2File(core, file,
                                        srcVertexOnRightColumn));
  fclose(file);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreWriteBinaryFile(HmlGraphCore const *core,
                            FILE               *file) {
  HML_ERR_PROLOGUE;
  HmlGraphCoreFileHeader header;
  size_t  numElements;
  uint32_t  numSrcVertices = core->maxSrcVertex - core->minSrcVertex + 1;

  memset(&header, 0, sizeof(HmlGraphCoreFileHeader));
  /* init graph-core file header */
  header.numEdges      = core->numEdges;
  header.minSrcVertex  = core->minSrcVertex;
  header.maxSrcVertex  = core->maxSrcVertex;
  header.minDestVertex = core->minDestVertex;
  header.maxDestVertex = core->maxDestVertex;
  numElements = fwrite(&header, sizeof(HmlGraphCoreFileHeader), 1, file);
  HML_ERR_GEN(numElements != 1, cHmlErrGeneral);
  /* + 1 below is to include the end of the last row */
  numElements = fwrite(core->R + core->minSrcVertex, sizeof(uint32_t),
                       numSrcVertices + 1, file);
  HML_ERR_GEN(numElements != numSrcVertices + 1, cHmlErrGeneral);
  numElements = fwrite(core->E, sizeof(uint32_t), core->numEdges, file);
  HML_ERR_GEN(numElements != core->numEdges, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreDefaultInsertIniter(HmlGraphCore *core, void *insertState) {
  (void)insertState;  /* avoid warning */
  return hmlGraphCoreAddEdgeInit(core);
}

/* this function only applies to bi-directional graph that is succinctly
 * represented by enforcing the constraint that (u, v) \in E iff u <= v.
 * However, sometimes we need to know the true bi-directional degree of
 * vertices in such a graph, and thus we need to count the degrees from those
 * (u, v) edges with u > v as well. This function does so and stores the
 * degree of each vertex in the D[] array, which must be allocated by
 * the caller and have a size of 'numVertices'
 */
HmlErrCode
hmlGraphCoreCountBidirectDegree(HmlGraphCore const *core,
                                uint32_t             *D,
                                uint32_t              numVertices) {
  HML_ERR_PROLOGUE;
  uint32_t  v;
  uint32_t *R = core->R;
  uint32_t *E = core->E;
  uint32_t *e;

  HML_ERR_GEN(!D, cHmlErrGeneral);
  HML_ERR_GEN(core->maxSrcVertex >= numVertices, cHmlErrGeneral);
  HML_ERR_GEN(core->maxDestVertex >= numVertices, cHmlErrGeneral);
  memset(D, 0, sizeof(uint32_t) * numVertices);
  for(v = core->minSrcVertex; v <= core->maxSrcVertex; ++v) {
    D[v] = (uint32_t)(R[v + 1] - R[v]);
    for(e = &E[R[v]]; e < &E[R[v + 1]]; e++) {
      HML_ERR_GEN(v > *e, cHmlErrGeneral);
      HML_ERR_GEN(*e >= numVertices, cHmlErrGeneral);
      D[*e]++;
    }
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreInsertFromSrc(HmlGraphCore               *core,
                          HmlGraphCore const         *src,
                          HmlGraphCoreInsertIniter    initer,
                          HmlGraphCoreInserter        inserter,
                          HmlGraphCoreInsertFinalizer finalizer,
                          void                       *insertState) {
  HML_ERR_PROLOGUE;
  uint32_t  v;
  uint32_t *R = src->R;
  uint32_t *E = src->E;
  uint32_t *e;

  if(initer) {
    HML_ERR_PASS(initer(core, insertState));
  }
  for(v = src->minSrcVertex; v <= src->maxSrcVertex; v++) {
    for(e = &E[R[v]]; e < &E[R[v + 1]]; e++) {
      HML_ERR_PASS(inserter(core, insertState, v, *e));
    }
  }
  if(finalizer) {
    HML_ERR_PASS(finalizer(core, insertState));
  }

  HML_NORMAL_RETURN;
}

/* count the degree by only allowing edge (u, v) \in E iff
 * P[u] < P[v] and the degree is stored in D, for
 * each vertex in the (bi-directional) graph.
 * Both P and D must be allocated by the caller and have the same
 * size of 'numVertices'
 */
HmlErrCode
hmlGraphCoreCountDegreeIfSmallerP(HmlGraphCore const *core,
                                  uint32_t const       *P,
                                  uint32_t             *D,
                                  uint32_t              numVertices) {
  HML_ERR_PROLOGUE;
  uint32_t  v;
  uint32_t *R = core->R;
  uint32_t *E = core->E;
  uint32_t *e;
  uint32_t  vP;
  uint32_t  eP;

  HML_ERR_GEN(!P || !D, cHmlErrGeneral);
  HML_ERR_GEN(core->maxSrcVertex >= numVertices, cHmlErrGeneral);
  HML_ERR_GEN(core->maxDestVertex >= numVertices, cHmlErrGeneral);
  memset(D, 0, sizeof(uint32_t) * numVertices);
  for(v = core->minSrcVertex; v <= core->maxSrcVertex; v++) {
    for(e = &E[R[v]]; e < &E[R[v + 1]]; e++) {
      HML_ERR_GEN(*e >= numVertices, cHmlErrGeneral);
      vP = P[v];
      eP = P[*e];
      HML_ERR_GEN(vP == eP, cHmlErrGeneral);
      if(vP < eP) {
        D[vP]++;
      }
      else {
        D[eP]++;
      }
    }
  }

  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlGraphCoreThreadFunc(uint32_t thread, void *threadArgs) {
  HML_ERR_PROLOGUE;
  HmlGraphCoreThreadArgs *args = (HmlGraphCoreThreadArgs *)threadArgs;
  HmlGraphCore           *core = args->core;
  uint32_t                minVertex;
  uint32_t                maxVertex;
  HML_ERR_PASS(hmlVertexPartitionGetMinMaxVertex(args->partition, thread,
               &minVertex, &maxVertex));
  return args->func(core, thread, minVertex, maxVertex, args->args);
}

HmlErrCode
hmlGraphCoreRunParaFunc(HmlGraphCore          *core,
                        HmlGraphCoreParaFunc   func,
                        void                  *args,
                        uint32_t               numThreads) {
  HML_ERR_PROLOGUE;
  HmlThreadGroup         threads;
  HmlGraphCoreThreadArgs threadArgs;
  HmlVertexPartition     vertexPartition;
  uint64_t               maxNumEdgesPerPartition =
    (core->numEdges + numThreads - 1) / numThreads;

  maxNumEdgesPerPartition += core->maxOutDegree;
  HML_ERR_PASS(hmlGraphCoreVertexPartition(core,
               maxNumEdgesPerPartition, &vertexPartition));
  HML_ERR_PASS(hmlThreadGroupInit(&threads, numThreads));
  threadArgs.core      = core;
  threadArgs.partition = &vertexPartition;
  threadArgs.func      = func;
  threadArgs.args      = args;
  HML_ERR_PASS(hmlThreadGroupRun(&threads, hmlGraphCoreThreadFunc, &threadArgs));
  HML_ERR_PASS(hmlThreadGroupDelete(&threads));
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreRunParaFuncByPartition(HmlGraphCore          *core,
                                   HmlGraphCoreParaFunc   func,
                                   void                  *args,
                                   HmlVertexPartition    *partition) {
  HML_ERR_PROLOGUE;
  HmlThreadGroup         threads;
  HmlGraphCoreThreadArgs threadArgs;

  HML_ERR_PASS(hmlThreadGroupInit(&threads, partition->numPartitions));
  threadArgs.core      = core;
  threadArgs.partition = partition;
  threadArgs.func      = func;
  threadArgs.args      = args;
  HML_ERR_PASS(hmlThreadGroupRun(&threads, hmlGraphCoreThreadFunc, &threadArgs));
  HML_ERR_PASS(hmlThreadGroupDelete(&threads));

  HML_NORMAL_RETURN;
}

static int
hmlGraphCoreVertexCompare(const void *a, const void *b) {
  return *(const uint32_t *)a - *(const uint32_t *)b;
}

static HmlErrCode
hmlGraphCoreSortFunc(HmlGraphCore     *core,
                     uint32_t          thread,
                     uint32_t          minVertex,
                     uint32_t          maxVertex,
                     void             *args) {
  HML_ERR_PROLOGUE;
  uint32_t  v;
  uint32_t *R = core->R;
  uint32_t *E = core->E;

  (void)thread; /* avoid warning */
  (void)args; /* avoid warning */
  minVertex = max(minVertex, core->minSrcVertex);
  maxVertex = min(maxVertex, core->maxSrcVertex);
  for(v = minVertex; v <= maxVertex; v++) {
    qsort(&E[R[v]], R[v + 1] - R[v], sizeof(uint32_t), hmlGraphCoreVertexCompare);
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreSortEdges(HmlGraphCore *core, uint32_t numThreads) {
  HML_ERR_PROLOGUE;
  uint32_t  v;
  uint32_t *R = core->R;
  uint32_t *E = core->E;

  if(numThreads <= 1) {
    for(v = core->minSrcVertex; v <= core->maxSrcVertex; v++) {
      qsort(&E[R[v]], R[v + 1] - R[v], sizeof(uint32_t), hmlGraphCoreVertexCompare);
    }
  }
  else {
    HML_ERR_PASS(hmlGraphCoreRunParaFunc(core, hmlGraphCoreSortFunc, NULL,
                                         numThreads));
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreSortEdgesByPartition(HmlGraphCore       *core,
                                 HmlVertexPartition *partition) {
  HML_ERR_PROLOGUE;
  uint32_t  v;
  uint32_t *R = core->R;
  uint32_t *E = core->E;

  if(partition->numPartitions <= 1) {
    for(v = core->minSrcVertex; v <= core->maxSrcVertex; v++) {
      qsort(&E[R[v]], R[v + 1] - R[v], sizeof(uint32_t), hmlGraphCoreVertexCompare);
    }
  }
  else {
    HML_ERR_PASS(hmlGraphCoreRunParaFuncByPartition(core, hmlGraphCoreSortFunc,
                 NULL, partition));
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreReadBinaryFile(HmlGraphCore *core,
                           FILE         *file) {
  HML_ERR_PROLOGUE;
  HmlGraphCoreFileHeader header;
  size_t  numElements;

  memset(core, 0, sizeof(HmlGraphCore));
  numElements = fread(&header, sizeof(HmlGraphCoreFileHeader), 1, file);
  HML_ERR_GEN(numElements != 1, cHmlErrGeneral);
  core->numEdges          = header.numEdges;
  core->minSrcVertex      = header.minSrcVertex;
  core->maxSrcVertex      = header.maxSrcVertex;
  core->minDestVertex     = header.minDestVertex;
  core->maxDestVertex     = header.maxDestVertex;
  core->maxNumSrcVertices = header.maxSrcVertex - header.minSrcVertex + 1;
  core->minMinSrcVertex   = header.minSrcVertex;
  /* allocate memory for the row indices */
  MALLOC(core->R0, uint32_t, core->maxNumSrcVertices + 1);
  /* shift R such that &R[minSrcVertex] == &R0[0] */
  core->R = core->R0 - core->minSrcVertex;
  /* + 1 below is to include the end of the last row */
  numElements =
    fread(core->R0, sizeof(uint32_t), core->maxNumSrcVertices + 1, file);
  HML_ERR_GEN(numElements != core->maxNumSrcVertices + 1, cHmlErrGeneral);
  /* allocate memory for the edges */
  MALLOC(core->E, uint32_t, core->numEdges);
  numElements = fread(core->E, sizeof(uint32_t), core->numEdges, file);
  HML_ERR_GEN(numElements != core->numEdges, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreWriteTsv2File(HmlGraphCore const *core,
                          FILE               *file,
                          bool                srcVertexOnRightColumn) {
  HML_ERR_PROLOGUE;
  uint32_t        maxSrcVertex = core->maxSrcVertex;
  uint32_t       *R = core->R;
  uint32_t       *E = core->E;
  uint32_t       *e;
  uint32_t        v;

  if(!srcVertexOnRightColumn) {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
        fprintf(file, "%u %u\n", v, *e);
      }
    }
  }
  else {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
        fprintf(file, "%u %u\n", *e, v);
      }
    }
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreWriteVertexInfoTsv3File(HmlGraphCore const *core,
                                    FILE               *file,
                                    bool                srcVertexOnRightColumn,
                                    uint32_t             *vertexInfoArr,
                                    bool                destVertexInfo) {
  HML_ERR_PROLOGUE;
  uint32_t        maxSrcVertex = core->maxSrcVertex;
  uint32_t       *R = core->R;
  uint32_t       *E = core->E;
  uint32_t       *e;
  uint32_t        v;
  uint32_t        srcVertexInfo;

  if(!srcVertexOnRightColumn) {
    if(!destVertexInfo) {
      for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
        srcVertexInfo = vertexInfoArr[v];
        for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
          fprintf(file, "%u %u %u\n", v, srcVertexInfo, *e);
        }
      }
    }
    else {
      for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
        for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
          fprintf(file, "%u %u %u\n", v, vertexInfoArr[*e], *e);
        }
      }
    }
  }
  else if(!destVertexInfo) {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      srcVertexInfo = vertexInfoArr[v];
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
        fprintf(file, "%u %u %u\n", *e, srcVertexInfo, v);
      }
    }
  }
  else {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
        fprintf(file, "%u %u %u\n", *e, vertexInfoArr[*e], v);
      }
    }
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreWriteVertexInfoTsv4File(HmlGraphCore const *core,
                                    FILE               *file,
                                    bool                srcVertexOnRightColumn,
                                    uint16_t              edgeType,
                                    uint32_t             *vertexInfoArr,
                                    bool                destVertexInfo) {
  HML_ERR_PROLOGUE;
  uint32_t        maxSrcVertex = core->maxSrcVertex;
  uint32_t       *R = core->R;
  uint32_t       *E = core->E;
  uint32_t       *e;
  uint32_t        v;
  uint32_t        srcVertexInfo;

  if(!srcVertexOnRightColumn) {
    if(!destVertexInfo) {
      for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
        srcVertexInfo = vertexInfoArr[v];
        for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
          fprintf(file, "%u %hu %u %u\n", v, edgeType, srcVertexInfo, *e);
        }
      }
    }
    else {
      for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
        for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
          fprintf(file, "%u %hu %u %u\n", v, edgeType, vertexInfoArr[*e], *e);
        }
      }
    }
  }
  else if(!destVertexInfo) {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      srcVertexInfo = vertexInfoArr[v];
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
        fprintf(file, "%u %hu %u %u\n", *e, edgeType, srcVertexInfo, v);
      }
    }
  }
  else {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e) {
        fprintf(file, "%u %hu %u %u\n", *e, edgeType, vertexInfoArr[*e], v);
      }
    }
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreDeleteD(HmlGraphCore *core) {
  HML_ERR_PROLOGUE;

  HML_ERR_GEN(core->D && core->D + core->minMinSrcVertex != core->D0,
              cHmlErrGeneral);
  FREE(core->D0);
  core->D = NULL;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreWriteEdgeInfoTsv3File(HmlGraphCore const *core,
                                  FILE               *file,
                                  bool                srcVertexOnRightColumn,
                                  uint32_t             *edgeInfoArr) {
  HML_ERR_PROLOGUE;
  uint32_t        maxSrcVertex = core->maxSrcVertex;
  uint32_t       *R = core->R;
  uint32_t       *E = core->E;
  uint32_t       *e;
  uint32_t        v;
  uint32_t        numEdges = 0;

  if(!srcVertexOnRightColumn) {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e, ++numEdges) {
        fprintf(file, "%u %u %u\n", v, edgeInfoArr[numEdges], *e);
      }
    }
  }
  else {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e, ++numEdges) {
        fprintf(file, "%u %u %u\n", *e, edgeInfoArr[numEdges], v);
      }
    }
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreWriteEdgeInfoTsv4File(HmlGraphCore const *core,
                                  FILE               *file,
                                  bool                srcVertexOnRightColumn,
                                  uint16_t              edgeType,
                                  uint32_t             *edgeInfoArr) {
  HML_ERR_PROLOGUE;
  uint32_t        maxSrcVertex = core->maxSrcVertex;
  uint32_t       *R = core->R;
  uint32_t       *E = core->E;
  uint32_t       *e;
  uint32_t        v;
  uint32_t        numEdges = 0;

  if(!srcVertexOnRightColumn) {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e, ++numEdges) {
        fprintf(file, "%u %hu %u %u\n", v, edgeType, edgeInfoArr[numEdges], *e);
      }
    }
  }
  else {
    for(v = core->minSrcVertex; v <= maxSrcVertex; ++v) {
      for(e = &E[R[v]]; e < &E[R[v+1]]; ++e, ++numEdges) {
        fprintf(file, "%u %hu %u %u\n", *e, edgeType, edgeInfoArr[numEdges], v);
      }
    }
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreBindTexture(HmlGraphCore            *core,
                        const texture<uint32_t, 1> &texDataR,
                        const texture<uint32_t, 1> &texDataE) {
  HML_ERR_PROLOGUE;
  size_t texOffset;

  /* bind row data to texture */
  HANDLE_ERROR(cudaBindTexture(&texOffset, texDataR, core->R0,
                               sizeof(uint32_t) * core->maxNumSrcVertices));
  /* check for non-zero offset */
  if(texOffset != 0) {
    fprintf(stderr, "; Error: Row texture offset != 0\n");
    HML_ERR_GEN(texOffset, cHmlErrGeneral);
  }

  /* bind edge data to texture */
  HANDLE_ERROR(cudaBindTexture(&texOffset, texDataE, core->E,
                               sizeof(uint32_t) * core->numEdges));
  /* check for non-zero offset */
  if(texOffset != 0) {
    fprintf(stderr, "; Error: Edge texture offset != 0\n");
    HML_ERR_GEN(texOffset, cHmlErrGeneral);
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGraphCoreUnbindTexture(const texture<uint32_t, 1> &texDataR,
                          const texture<uint32_t, 1> &texDataE) {
  HML_ERR_PROLOGUE;
  HANDLE_ERROR(cudaUnbindTexture(texDataR));
  HANDLE_ERROR(cudaUnbindTexture(texDataE));

  HML_NORMAL_RETURN;
}
