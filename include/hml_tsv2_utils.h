/*****************************************************************
*  Copyright (c) 2014. Palo Alto Research Center                *
*  All rights reserved.                                         *
*****************************************************************/

/*
 * hml_tsv2_utils.h: prototypes for tsv2 utilities
 */
#ifndef HML_TSV2_UTILS_H_INCLUDED_
#define HML_TSV2_UTILS_H_INCLUDED_

#include "hml_common.h"

/* make prototypes usable from C++ */
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* reads a vertex degree file that has the following format:
 * --- beginning of file ---
 * <number of vertices>
 * <in-degree of vertex v_1> <out-degree of vertex v_1>
 * ...
 * <in-degree of vertex v_n> <out-degree of vertex v_n>
 * --- end of file ---
 * returns through pointers *inDegreeArr and *inDegreeArrSize
 * the in-degree array and size of the in-degree array;
 * the same is true for out-degree array and its size;
 * finally, *numEdges returns the total number of edges
 */
HmlErrCode
hmlTsv2InOutDegreeReadFile(FILE     *file,
                           uint32_t  **inDegreeArr,
                           uint32_t   *inDegreeArrSize,
                           uint32_t  **outDegreeArr,
                           uint32_t   *outDegreeArrSize,
                           uint64_t   *numEdges);

/* same as above, except that it takes a file name as argument */
HmlErrCode
hmlTsv2InOutDegreeReadFileWithName(char const *fileName,
                                   uint32_t    **inDegreeArr,
                                   uint32_t     *inDegreeArrSize,
                                   uint32_t    **outDegreeArr,
                                   uint32_t     *outDegreeArrSize,
                                   uint64_t     *numEdges);

/* counts the in- and out-degrees for all vertices.
 * returns through pointers *inDegreeArr and *inDegreeArrSize
 * the in-degree array and size of the in-degree array;
 * the same is true for out-degree array and its size;
 * finally, *numEdges returns the total number of edges
 */
HmlErrCode
hmlTsv2InOutDegreeCountFile(FILE         *file,
                            uint32_t      **inDegreeArr,
                            uint32_t       *inDegreeArrSize,
                            uint32_t      **outDegreeArr,
                            uint32_t       *outDegreeArrSize,
                            uint64_t       *numEdges);

/* same as above, except that it takes a file name as argument */
HmlErrCode
hmlTsv2InOutDegreeCountFileWithName(char const  *fileName,
                                    uint32_t      **inDegreeArr,
                                    uint32_t       *inDegreeArrSize,
                                    uint32_t      **outDegreeArr,
                                    uint32_t       *outDegreeArrSize,
                                    uint64_t       *numEdges);

#ifdef __cplusplus
}
#endif

#endif  /* HML_TSV2_UTILS_H_INCLUDED_ */
