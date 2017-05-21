/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_PAGERANK_SPMV_H_INCLUDED_
#define HML_PAGERANK_SPMV_H_INCLUDED_

#include "hml_common.h"
#include "hml_graph_core.h"

void
hmlGraphReadTsv4(char *filename, bool sortedBySubject, HmlGraph *graph);

void
hmlGraphPrintStats(FILE *file, HmlGraph *graph);

void
hmlGraphPrintEdges(FILE *file, HmlGraph *graph, bool sortedBySubject);

void
hmlGraphDeleteHost(HmlGraph *graph);

void
hmlPagerankSpmvCpu(HmlGraph   *hostGraph,
                   float       dampingFactor,
                   uint32_t    numIters,
                   uint32_t    printTopK,
                   const char *outFilenamePrefix,
                   const char *outFileNameExtension);

void
hmlPagerankSpmvGpu(HmlGraph   *hostGraph,
                   float       dampingFactor,
                   uint32_t    numIters,
                   uint32_t    printTopK,
                   const char *outFilenamePrefix,
                   const char *outFileNameExtension,
                   uint32_t    verbosity);

#endif /* HML_PAGERANK_SPMV_H_INCLUDED_ */
