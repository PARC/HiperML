/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_UTILS_H_INCLUDED_
#define HML_UTILS_H_INCLUDED_

#include "hml_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MALLOC(p,t,s) do {                                      \
    if ((p=(t*)malloc(sizeof(t)*(s))) == NULL) {                \
      fprintf(stderr, "Host memory failed in %s at line %d\n",  \
              __FILE__, __LINE__ );                             \
      exit( EXIT_FAILURE );}                                    \
  } while(0)

#define CALLOC(p,t,s) do {                                      \
    if ((p=(t*)calloc((s), sizeof(t))) == NULL) {               \
      fprintf(stderr, "Host memory failed in %s at line %d\n",  \
              __FILE__, __LINE__ );                             \
      exit( EXIT_FAILURE );}                                    \
  } while(0)

#define REALLOC(p,t,s) do {                                     \
    if ((p=(t*)realloc(p, sizeof(t)*(s))) == NULL) {            \
      fprintf(stderr, "Host memory failed in %s at line %d\n",  \
              __FILE__, __LINE__ );                             \
      exit( EXIT_FAILURE );}                                    \
  } while(0)

#define FREE(p) do {                            \
    if (p != NULL) {                            \
      free(p);                                  \
      p = NULL;}                                \
  } while(0)

#define HANDLE_ERROR( err ) (hmlHandleError( err, __FILE__, __LINE__ ))

typedef struct {
  double cpuTime;
  double wallTime;
} HmlTime;

HmlErrCode hmlGetSecs(double *cpuSecs, double *wallSecs);

HmlErrCode hmlGetTime(HmlTime *time);

/* return a randomly generated float */
float hmlRandomFloat();

FILE* openFile(const char *filename, const char *mode);

void hmlHandleError(cudaError_t err, const char *file, int line);

int hmlMaxNumRegistersPerThread(const cudaDeviceProp *prop);

uint32_t* hmlDeviceUint32ArrayAllocBind(int                numElems,
                                        texture<uint32_t, 1> &texFloatArr);

float* hmlDeviceFloatArrayAlloc(int numElems);

float* hmlDeviceFloatArrayAllocLoad(const float       *hostArr,
                                    int                numElems);

float* hmlDeviceFloatArrayAllocBind(int                numElems,
                                    texture<float, 1> &texFloatArr);

float* hmlDeviceFloatArrayAllocLoadBind(const float       *hostArr,
                                        int                numElems,
                                        texture<float, 1> &texFloatArr);

void hmlDevicePropertyCheck(const cudaDeviceProp *prop);

void hmlDevicePropertyPrint(const cudaDeviceProp *prop);

/* global variables */
extern cudaDeviceProp cudaActiveDeviceProp;

#ifdef __cplusplus
}
#endif

#endif /* HML_UTILS_H_INCLUDED_ */
