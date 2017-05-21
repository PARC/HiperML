/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#ifndef HML_TYPES_H_INCLUDED_
#define HML_TYPES_H_INCLUDED_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>
#include <stdint.h>

/* make prototypes usable from C++ */
#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

/*! This is the type returned by all HiperML procedures */
typedef unsigned int   HmlErrCode;

#ifdef __CUDACC__
typedef struct {
  dim3     grid;
  dim3     block;
  int      allocBytes;   /* dynamic shared memory size in bytes */
} HmlKernelArg;
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* HML_TYPES_H_INCLUDED_ */
