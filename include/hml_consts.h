#ifndef HML_CONSTS_H_INCLUDED_
#define HML_CONSTS_H_INCLUDED_

/* CUDA constants */
#define cHmlThreadsPerWarp     32
#define cHmlMaxThreadsPerBlock 1024
#define cHmlMaxSmemBytes       49152
#define cHmlMaxCmemBytes      65536
#define cHmlMaxBlockDimX       1024
#define cHmlMaxBlockDimY       1024
#define cHmlMaxBlockDimZ       64
#define cHmlMaxGridDimX        65535
#define cHmlMaxGridDimY        65535
#define cHmlMaxGridDimZ        65535

/* max 1d texture size in terms of # of elements, NOT in bytes */
#define cHmlMaxCudaTexture1DLinear (1 << 27)

/* constants used in #define macros */
#define cBytesPerFloat        4
#define cBytesPerInt          4
#define cBytesPerDouble       8

#define cHmlLineBufferSize   1024

#endif /* HML_CONSTS_H_INCLUDED_ */
