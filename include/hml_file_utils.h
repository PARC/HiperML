/*****************************************************************
 *  Copyright (c) 2015, Palo Alto Research Center.               *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_FILE_UTILS_H_INCLUDED_
#define HML_FILE_UTILS_H_INCLUDED_

/* this file contains the prototypes for the routines that are used to
 * perform file-related operations.
 */
#include <cuda.h>
#include "hml_types.h"

#define cHmlFileMagicStrMaxLen   128

/* make prototypes usable from C++ */
#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

/* On non-Windows OS, pathBufSize must be at least _POSIX_PATH_MAX,
 * It's recommended that pathBufSize to be >= 4096
 */
HmlErrCode
hmlFileFullPath(char const *fileName, char *path, uint32_t pathBufSize);

HmlErrCode
hmlGetCurrentDirectory(char *path, uint32_t pathBufSize);

/* open a file for read */
HmlErrCode
hmlFileOpenRead(const char  *fileName, FILE **file);

/* open a file for read and check if the first magicStrLen bytes
 * match magicStr
 */
HmlErrCode
hmlFileOpenReadWithMagicStr(char const *fileName,
                            char const *magicStr,
                            FILE      **file);

HmlErrCode
hmlFileOpenReadWithDefaultMagicStr(char const *fileName,
                                   char const *magicStrBase,
                                   FILE      **file);

/* open a file for write */
HmlErrCode
hmlFileOpenWrite(const char  *fileName, FILE **file);

HmlErrCode
hmlFileOpenWriteWithMagicStr(char const *fileName,
                             char const *magicStr,
                             FILE      **file);

HmlErrCode
hmlFileOpenWriteWithDefaultMagicStr(char const *fileName,
                                    char const *magicStrBase,
                                    FILE      **file);

/* close a previously opened file */
HmlErrCode
hmlFileClose(FILE *file);

char const *
hmlBasename(char const *path);

/********************************************************************
 * Set 'file' to binary mode
 * The routine is often used to set stdin or stdout on Windows to
 * binary mode so that line endings are always in Unix style.
 *
 * \param [in/out] file         pointer to FILE structure
 *********************************************************************/
HmlErrCode
hmlFileSetBinaryMode(FILE *file);

HmlErrCode
hmlFileCreateLineOffsetArray(FILE     *file,
                             uint64_t  **offsetArr,
                             uint32_t   *offsetArrSize);

HmlErrCode
hmlFileWriteUint32Array(FILE           *file,
                        const uint32_t   *array,
                        uint32_t          arraySize);

HmlErrCode
hmlFileWriteUint64Array(FILE           *file,
                        const uint64_t   *array,
                        uint32_t          arraySize);

HmlErrCode
hmlFileReadUint32Array(FILE           *file,
                       uint32_t        **array,
                       uint32_t         *arraySize);

HmlErrCode
hmlFileReadUint64Array(FILE           *file,
                       uint64_t        **array,
                       uint32_t         *arraySize);

HmlErrCode
hmlFilePrintFloatArrayExt(FILE          *file,
                          float const *array,
                          uint32_t         arraySize,
                          int            separator);

HmlErrCode
hmlFilePrintFloatArray(FILE          *file,
                       float const *array,
                       uint32_t         arraySize);

/* Caller is responsible for freeing array */
HmlErrCode
hmlFileScanFloatArrayExt(FILE     *file,
                         float **array,
                         uint32_t   *arraySize,
                         int       separator);

/* Caller is responsible for freeing array */
HmlErrCode
hmlFileScanFloatArray(FILE     *file,
                      float **array,
                      uint32_t   *arraySize);

/* Caller is responsible for freeing array */
HmlErrCode
hmlFilePathScanFloatArray(char const *path,
                          float   **array,
                          uint32_t     *arraySize);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif  /* HML_FILE_UTILS_H_INCLUDED_ */
