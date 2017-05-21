/*****************************************************************
 *  Copyright (c) 2014. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

/*
 *  hml_file_utils.c:
 */

#ifdef _WIN32
#include <Windows.h>
#include <io.h>
#endif

#ifndef _WIN32
#include <libgen.h> /* for basename() */
#include <unistd.h>
#endif
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include "hml_utils.h"
#include "hml_file_utils.h"
#include "hml_errmacros.h"

#define cHmlFileOffsetArrSizeInit 1024
#define cHmlFileOffsetArrSizeMax  ((uint32_t)-1)

/* On non-Windows OS, pathBufSize must be at least _POSIX_PATH_max,
 * It's recommended that pathBufSize to be >= 4096
 */
HmlErrCode
hmlFileFullPath(char const *fileName, char *path, uint32_t pathBufSize) {
  HML_ERR_PROLOGUE;
#ifndef _WIN32
  char *retptr;
  HML_ERR_GEN(pathBufSize < _POSIX_PATH_MAX, cHmlErrGeneral);
  retptr = realpath(fileName, path);
  HML_ERR_GEN(!retptr, cHmlErrGeneral);
#else
  DWORD retval = GetLongPathNameA(fileName, path, pathBufSize);
  HML_ERR_GEN(!retval, cHmlErrGeneral);
#endif
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlGetCurrentDirectory(char *path, uint32_t pathBufSize) {
  HML_ERR_PROLOGUE;
#ifndef _WIN32
  char *retptr;
  retptr = getcwd(path, pathBufSize);
  HML_ERR_GEN(!retptr, cHmlErrGeneral);
#else
  DWORD retval = GetCurrentDirectoryA(pathBufSize, path);
  HML_ERR_GEN(!retval, cHmlErrGeneral);
#endif
  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFileOpenRead(char const *fileName, FILE **file) {
  HML_ERR_PROLOGUE;
  char pwd[4098];
  char errMsg[4098];

  *file = fopen(fileName, "rb");
  if(!*file) {
    HML_ERR_PASS(hmlGetCurrentDirectory(pwd, sizeof(pwd)));
    hmlSnprintf(errMsg, sizeof(errMsg), "%s (pwd: %s)", fileName, pwd);
    HML_ERR_GEN_EXT(!*file, cHmlErrGeneral, errMsg);
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFileOpenWrite(char const *fileName, FILE **file) {
  HML_ERR_PROLOGUE;
  char pwd[4098];
  char errMsg[4098];

  *file = fopen(fileName, "wb");
  if(!*file) {
    HML_ERR_PASS(hmlGetCurrentDirectory(pwd, sizeof(pwd)));
    hmlSnprintf(errMsg, sizeof(errMsg), "%s (pwd: %s)", fileName, pwd);
    HML_ERR_GEN_EXT(!*file, cHmlErrGeneral, errMsg);
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFileClose(FILE *file) {
  HML_ERR_PROLOGUE;
  int32_t result;

  result = fclose(file);
  HML_ERR_GEN(result, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

char const *
hmlBasename(char const *path) {
  if(!path) return NULL;
  char const *s = strrchr(path, '/');
  return (s == NULL) ? path : ++s;
}

HmlErrCode
hmlFileSetBinaryMode(FILE *file) {
  HML_ERR_PROLOGUE;

#ifdef _WIN32
  int result = _setmode(_fileno(file), _O_BINARY);
  HML_ERR_GEN(result == -1, cHmlErrGeneral);
#else
  /* avoid warning on Linux */
  (void)file;
#endif

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFileCreateLineOffsetArray(FILE     *file,
                             uint64_t  **offsetArr,
                             uint32_t   *offsetArrSize) {
  HML_ERR_PROLOGUE;
  uint32_t   line = 0;
  int      ch;

  HML_ERR_GEN(!offsetArr, cHmlErrGeneral);
  HML_ERR_GEN(!offsetArrSize, cHmlErrGeneral);
  if(*offsetArr == NULL) {
    MALLOC(*offsetArr, uint64_t, cHmlFileOffsetArrSizeInit);
    *offsetArrSize = cHmlFileOffsetArrSizeInit;
  }
  /* reset to the beginning of the file */
  rewind(file);
  while(!feof(file)) {
    if(*offsetArrSize <= line) {
      if(*offsetArrSize < cHmlFileOffsetArrSizeMax / 2) {
        *offsetArrSize *= 2;
        *offsetArrSize = max(*offsetArrSize, cHmlFileOffsetArrSizeInit);
      } else {
        *offsetArrSize = cHmlFileOffsetArrSizeMax;
      }
      REALLOC(*offsetArr, uint64_t, *offsetArrSize);
    }
    HML_ERR_GEN(line >= *offsetArrSize, cHmlErrGeneral);
    (*offsetArr)[line] = ftell(file);
    line++;
    while((ch = fgetc(file)) != '\n' && ch != EOF);
  }
  REALLOC(*offsetArr, uint64_t, line);
  *offsetArrSize = line;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFileWriteUint32Array(FILE *file, uint32_t const *array, uint32_t arraySize) {
  HML_ERR_PROLOGUE;
  size_t   result;

  HML_ERR_GEN(!array, cHmlErrGeneral);
  result = fwrite(array, sizeof(uint32_t), arraySize, file);
  HML_ERR_GEN(result != (size_t)arraySize, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFileWriteUint64Array(FILE *file, uint64_t const *array, uint32_t arraySize) {
  HML_ERR_PROLOGUE;
  size_t   result;

  HML_ERR_GEN(!array, cHmlErrGeneral);
  result = fwrite(array, sizeof(uint64_t), arraySize, file);
  HML_ERR_GEN(result != (size_t)arraySize, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFileReadUint32Array(FILE *file, uint32_t **array, uint32_t *arraySize) {
  HML_ERR_PROLOGUE;
  size_t   result;
  uint64_t   size;

  HML_ERR_GEN(!array, cHmlErrGeneral);
  fseek(file, 0, SEEK_END);
  size = ftell(file);
  *arraySize = (uint32_t)(size / sizeof(uint32_t));
  HML_ERR_GEN(*arraySize * sizeof(uint32_t) != size, cHmlErrGeneral);
  REALLOC(*array, uint32_t, *arraySize);
  rewind(file);
  result = fread(*array, sizeof(uint32_t), *arraySize, file);
  HML_ERR_GEN(result != (size_t)*arraySize, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFileReadUint64Array(FILE *file, uint64_t **array, uint32_t *arraySize) {
  HML_ERR_PROLOGUE;
  size_t   result;
  uint64_t   size;

  HML_ERR_GEN(!array, cHmlErrGeneral);
  fseek(file, 0, SEEK_END);
  size = ftell(file);
  *arraySize = (uint32_t)(size / sizeof(uint64_t));
  HML_ERR_GEN(*arraySize * sizeof(uint64_t) != size, cHmlErrGeneral);
  REALLOC(*array, uint64_t, *arraySize);
  rewind(file);
  result = fread(*array, sizeof(uint64_t), *arraySize, file);
  HML_ERR_GEN(result != (size_t)*arraySize, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFilePrintfloatArrayExt(FILE          *file,
                            float const *array,
                            uint32_t         arraySize,
                            int            separator) {
  HML_ERR_PROLOGUE;
  uint32_t idx;
  int    result;

  HML_ERR_GEN(!array || !arraySize, cHmlErrGeneral);
  for(idx = 0; idx < arraySize; idx++) {
    result = fprintf(file, "%.9g", array[idx]);
    HML_ERR_GEN(result < 0, cHmlErrGeneral);
    fputc(separator, file);
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlFilePrintfloatArray(FILE          *file,
                         float const *array,
                         uint32_t         arraySize) {
  return hmlFilePrintfloatArrayExt(file, array, arraySize, '\n');
}
