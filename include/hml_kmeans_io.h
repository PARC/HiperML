/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_KMEANS_IO_H_INCLUDED_
#define HML_KMEANS_IO_H_INCLUDED_

#include "hml_common.h"

#define cHmlLineBufferSize 1024

#ifdef __cplusplus
extern "C" {
#endif

void hmlKmeansReadInputFile(const char  *fileName,
                            float    **ppData,
                            uint32_t      *pNumColumns,
                            uint32_t      *pNumRows);

#ifdef __cplusplus
}
#endif

#endif /* HML_KMEANS_IO_H_INCLUDED_*/
