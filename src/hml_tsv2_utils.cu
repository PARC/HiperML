/*****************************************************************
*  Copyright (c) 2017. Palo Alto Research Center                *
*  All rights reserved.                                         *
*****************************************************************/

/*
 * hml_tsv2_utils.c
 */
#include "hml_tsv2_utils.h"
#include "hml_file_utils.h"

#define cHmlTsv2UtilsBufferSizeMax           1024
#define cHmlTsv2UtilsNumVerticesInit         1000000
#define cHmlTsv2UtilsNumVerticesMax          0xFFFFFFFF

HmlErrCode
hmlTsv2InOutDegreeReadFile(
  FILE     *file,
  uint32_t  **inDegreeArr,
  uint32_t   *inDegreeArrSize,
  uint32_t  **outDegreeArr,
  uint32_t   *outDegreeArrSize,
  uint64_t   *numEdges) {
  HML_ERR_PROLOGUE;
  int32_t   result;
  uint32_t  vertex;
  uint32_t *outDegreeArray = NULL;
  uint32_t  srcVertexMax = 0;
  uint32_t *inDegreeArray = NULL;
  uint32_t  destVertexMax = 0;
  uint32_t  digit;
  uint32_t  outDegree;
  uint32_t  inDegree;
  uint32_t  numVertices;
  char    line[cHmlTsv2UtilsBufferSizeMax];
  char   *str;
  uint64_t  degreeSum = 0;

  result = fscanf(file, "%u\n", &numVertices);
  HML_ERR_GEN(result != 1, cHmlErrGeneral);
  if(outDegreeArr) {
    outDegreeArray = *outDegreeArr;
    HML_ERR_GEN(!outDegreeArrSize, cHmlErrGeneral);
    if(!outDegreeArray) {
      MALLOC(outDegreeArray, uint32_t, numVertices);
    }
    else {
      REALLOC(outDegreeArray, uint32_t, numVertices);
    }
  }
  if(inDegreeArr) {
    inDegreeArray = *inDegreeArr;
    HML_ERR_GEN(!inDegreeArrSize, cHmlErrGeneral);
    if(!inDegreeArray) {
      MALLOC(inDegreeArray, uint32_t, numVertices);
    }
    else {
      REALLOC(inDegreeArray, uint32_t, numVertices);
    }
  }
  for(vertex = 0; vertex < numVertices; ++vertex) {
    /* Go get the next <outDegree, inDegree> pair, if it exists. */
    str = fgets(line, cHmlTsv2UtilsBufferSizeMax, file);
    if(!str) {
      break;
    }
    /* the four while loops below achieve the same
     * function as:
     * fscanf(file, "%u %u\n", &inDegree, &outDegree);
     */
    inDegree = 0;
    while((digit = (uint32_t)*str++) != ' ' && digit != '\t') {
      inDegree = inDegree * 10 + digit - '0';
    }
    degreeSum += inDegree;
    if(inDegreeArray) {
      inDegreeArray[vertex] = inDegree;
      if(inDegree) {
        destVertexMax = vertex;
      }
    }
    outDegree = 0;
    while((digit = (uint32_t)*str++) != '\n') {
      outDegree = outDegree * 10 + digit - '0';
    }
    degreeSum += outDegree;
    if(outDegreeArray) {
      outDegreeArray[vertex] = outDegree;
      if(outDegree) {
        srcVertexMax = vertex;
      }
    }
    if(feof(file)) {
      break;
    }
  }

  if(inDegreeArray) {
    *inDegreeArrSize = destVertexMax + 1;
    if(*inDegreeArrSize != numVertices) {
      REALLOC(inDegreeArray, uint32_t, *inDegreeArrSize);
    }
    *inDegreeArr = inDegreeArray;
  }
  if(outDegreeArray) {
    *outDegreeArrSize = srcVertexMax + 1;
    if(*outDegreeArrSize != numVertices) {
      REALLOC(outDegreeArray, uint32_t, *outDegreeArrSize);
    }
    *outDegreeArr = outDegreeArray;
  }
  if(numEdges) {
    *numEdges = degreeSum / 2;
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTsv2InOutDegreeReadFileWithName(
  const char *fileName,
  uint32_t    **inDegreeArr,
  uint32_t     *inDegreeArrSize,
  uint32_t    **outDegreeArr,
  uint32_t     *outDegreeArrSize,
  uint64_t     *numEdges) {
  HML_ERR_PROLOGUE;
  FILE      *file;

  HML_ERR_PASS(hmlFileOpenRead(fileName, &file));
  HML_ERR_PASS(hmlTsv2InOutDegreeReadFile(
                 file,
                 inDegreeArr,
                 inDegreeArrSize,
                 outDegreeArr,
                 outDegreeArrSize,
                 numEdges));
  fclose(file);

  HML_NORMAL_RETURN;
}

/* *numVertices holds the initial size of numPredecessorsArr
 * supplied by the caller, and is subject to modification, if the
 * initial size of numPredecessorsArr is too small to hold
 * all the destination vertices. Thus, numVertices is both
 * an input and output argument, and upon return it holds the
 * actual number of destination vertices.
 */
HmlErrCode
hmlTsv2InOutDegreeCountFile(
  FILE         *file,
  uint32_t      **inDegreeArr,
  uint32_t       *inDegreeArrSize,
  uint32_t      **outDegreeArr,
  uint32_t       *outDegreeArrSize,
  uint64_t       *numEdges) {
  HML_ERR_PROLOGUE;
  uint64_t  numPairs = 0;
  uint32_t  digit;
  uint32_t  srcVertex;
  uint32_t  destVertex;
  uint32_t  srcVertexMax = 0;
  uint32_t  destVertexMax = 0;
  char    line[cHmlTsv2UtilsBufferSizeMax];
  char   *str;
  uint32_t *outDegreeArray = NULL;
  uint32_t  outDegreeArraySize = 0;
  uint32_t *inDegreeArray = NULL;
  uint32_t  inDegreeArraySize = 0;

  if(outDegreeArr) {
    outDegreeArray = *outDegreeArr;
    HML_ERR_GEN(!outDegreeArrSize, cHmlErrGeneral);
    outDegreeArraySize = *outDegreeArrSize;
    if(!outDegreeArray) {
      CALLOC(outDegreeArray, uint32_t, cHmlTsv2UtilsNumVerticesInit);
      outDegreeArraySize = cHmlTsv2UtilsNumVerticesInit;
    }
    else {
      memset(outDegreeArray, 0, sizeof(uint32_t) * outDegreeArraySize);
    }
  }
  if(inDegreeArr) {
    inDegreeArray = *inDegreeArr;
    HML_ERR_GEN(!inDegreeArrSize, cHmlErrGeneral);
    inDegreeArraySize = *inDegreeArrSize;
    if(!inDegreeArray) {
      CALLOC(inDegreeArray, uint32_t, cHmlTsv2UtilsNumVerticesInit);
      inDegreeArraySize = cHmlTsv2UtilsNumVerticesInit;
    }
    else {
      memset(inDegreeArray, 0, sizeof(uint32_t) * inDegreeArraySize);
    }
  }
  for(;;) {
    /* Go get the next <srcVertex, destVertex> tuple
     * if it exists.
     */
    str = fgets(line, cHmlTsv2UtilsBufferSizeMax, file);
    if(!str) {
      break;
    }
    /* the four while loops below achieve the same
     * function as:
     * fscanf(file, "%u %u\n", &srcVertex, &destVertex);
     */
    srcVertex = 0;
    while((digit = (uint32_t)*str++) != ' ' && digit != '\t') {
      srcVertex = srcVertex * 10 + digit - '0';
    }
    if(outDegreeArray) {
      srcVertexMax = max(srcVertexMax, srcVertex);
      if(srcVertex >= outDegreeArraySize) {
        uint64_t outDegreeArraySizePrev = outDegreeArraySize;
        outDegreeArraySize = max(outDegreeArraySize * 2, srcVertex + 1);
        outDegreeArraySize = min(outDegreeArraySize, cHmlTsv2UtilsNumVerticesMax);
        HML_ERR_GEN(srcVertex >= outDegreeArraySize, cHmlErrGeneral);
        REALLOC(outDegreeArray, uint32_t, outDegreeArraySize);
        memset(&outDegreeArray[outDegreeArraySizePrev], 0,
               sizeof(uint32_t) * (outDegreeArraySize - outDegreeArraySizePrev));
      }
      outDegreeArray[srcVertex]++;
    }
    destVertex = 0;
    while((digit = (uint32_t)*str++) != '\n' && digit != '\r') {
      destVertex = destVertex * 10 + digit - '0';
    }
    if(inDegreeArray) {
      destVertexMax = max(destVertexMax, destVertex);
      if(destVertex >= inDegreeArraySize) {
        uint64_t inDegreeArraySizePrev = inDegreeArraySize;
        inDegreeArraySize = max(inDegreeArraySize * 2, destVertex + 1);
        inDegreeArraySize = min(inDegreeArraySize, cHmlTsv2UtilsNumVerticesMax);
        HML_ERR_GEN(destVertex >= inDegreeArraySize, cHmlErrGeneral);
        REALLOC(inDegreeArray, uint32_t, inDegreeArraySize);
        memset(&inDegreeArray[inDegreeArraySizePrev], 0,
               sizeof(uint32_t) * (inDegreeArraySize - inDegreeArraySizePrev));
      }
      inDegreeArray[destVertex]++;
    }

    ++numPairs;
    if(feof(file)) {
      break;
    }
  }
  if(outDegreeArr) {
    REALLOC(outDegreeArray, uint32_t, srcVertexMax + 1);
    *outDegreeArr = outDegreeArray;
    *outDegreeArrSize = srcVertexMax + 1;
  }
  if(inDegreeArr) {
    REALLOC(inDegreeArray, uint32_t, destVertexMax + 1);
    *inDegreeArr = inDegreeArray;
    *inDegreeArrSize = destVertexMax + 1;
  }
  if(numEdges) {
    *numEdges = numPairs;
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlTsv2InOutDegreeCountFileWithName(
  const char   *fileName,
  uint32_t      **inDegreeArr,
  uint32_t       *inDegreeArrSize,
  uint32_t      **outDegreeArr,
  uint32_t       *outDegreeArrSize,
  uint64_t       *numEdges) {
  HML_ERR_PROLOGUE;
  FILE      *file;

  HML_ERR_PASS(hmlFileOpenRead(fileName, &file));
  HML_ERR_PASS(hmlTsv2InOutDegreeCountFile(
                 file,
                 inDegreeArr,
                 inDegreeArrSize,
                 outDegreeArr,
                 outDegreeArrSize,
                 numEdges));
  fclose(file);

  HML_NORMAL_RETURN;
}
