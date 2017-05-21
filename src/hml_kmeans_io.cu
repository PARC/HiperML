/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_kmeans_io.h"

/* k-means input file constants */
#define cHmlKmeansInitDataSize       1024
#define cHmlLineBufferSize   1024
#define cHmlKmeansMaxDenomValInt32   1000000000

void
hmlKmeansReadInputFile(const char  *fileName,
                       float    **ppData,
                       uint32_t      *pNumColumns,
                       uint32_t      *pNumRows) {
  FILE       *file;
  float    *pData;
  char       *pChar;
  uint32_t      charInt;
  char        line[cHmlLineBufferSize];
  bool        positive;
  int32_t       intVal;
  int32_t       numVal;
  int32_t       denomVal;
  float     cellVal;
  float    *pCurData;
  float    *pEndData;
  uint32_t      allocSizeofData;
  uint32_t      column;
  uint32_t      numColumns = (uint32_t)-1;
  uint32_t      numRows = 0;

  file = fopen(fileName, "rb");
  if(!file) {
    fprintf(stderr, "Cannot open file: %s\n", fileName);
    exit(EXIT_FAILURE);
  }
  MALLOC(pData, float, cHmlKmeansInitDataSize);
  pCurData = pData;
  allocSizeofData = cHmlKmeansInitDataSize;
  pEndData = &pData[allocSizeofData];
  while(true) {
    pChar = fgets(line, cHmlLineBufferSize, file);
    if(!pChar) {
      break;
    }
    column = 0;
    while(*pChar != '\n') {
      /* eat while spaces */
      while(*pChar == ' ' || *pChar == '\t') {
        ++pChar;
      }
      if(*pChar >= '0' && *pChar <= '9') {
        positive = true;
      }
      else if(*pChar == '-') {
        positive = false;
        ++pChar;
      }
      else if(*pChar == '+') {
        positive = true;
        ++pChar;
      }
      else {
        fprintf(stderr, "Err: invalid input line #%d: %s\n", numRows, line);
        exit(EXIT_FAILURE);
      }
      intVal = 0;
      while((charInt = (uint32_t)*pChar++) >= '0' && charInt <= '9') {
        intVal = intVal * 10 + charInt - '0';
        if(intVal < 0) {
          fprintf(stderr,
                  "Err: data at row %d column %d exceeds precision limits\n",
                  numRows + 1, column + 1);
          exit(EXIT_FAILURE);
        }
      }
      if(charInt == '.') {
        numVal = 0;
        denomVal = 1;
        while((charInt = (uint32_t)*pChar++) >= '0' && charInt <= '9') {
          if(denomVal >= cHmlKmeansMaxDenomValInt32) {
            fprintf(stderr,
                    "Err: data at row %d column %d exceeds precision limits:\n",
                    numRows + 1, column + 1);
            fprintf(stderr, "     numerator = %d..., denominator = %d...\n",
                    numVal, denomVal);
            exit(EXIT_FAILURE);
          }
          numVal = numVal * 10 + charInt - '0';
          denomVal *= 10;
        }
        cellVal = (positive)? (float)numVal / (float)denomVal :
                  -(float)numVal / (float)denomVal;
      }
      else {
        cellVal = 0.0;
      }
      cellVal += (positive? (float)intVal : -(float)intVal);
      if(pCurData >= pEndData) {
        REALLOC(pData, float, allocSizeofData * 2);
        pCurData = &pData[allocSizeofData];
        allocSizeofData *= 2;
        pEndData = &pData[allocSizeofData];
      }
      *pCurData++ = cellVal;
      ++column;
      if(charInt == '\n') {
        break;
      }
    }
    ++numRows;
    if(numColumns == (uint32_t)-1) {
      numColumns = column;
    }
    if(column != numColumns) {
      fprintf(stderr,
              "Err: row %d has %d columns, while the previous row has %d\n",
              numRows, column, numColumns);
      exit(EXIT_FAILURE);
    }
  }
  fclose(file);
  if(column > 0 && column != numColumns) {
    fprintf(stderr,
            "Err: row %d has %d columns, while the previous row has %d\n",
            numRows, column, numColumns);
    exit(EXIT_FAILURE);
  }
  REALLOC(pData, float, numColumns * numRows);
  *ppData = pData;
  *pNumColumns = numColumns;
  *pNumRows = numRows;
}
