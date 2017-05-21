/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include "hml_common.h"

#ifdef WIN32
#include <Windows.h>
#include <time.h>
#else
#include <unistd.h>
#include <sys/time.h>
#include <sys/times.h>
#endif

/* return the CPU and wall-clock seconds through
 * two pointers 'cpuSecs' and 'wallSecs' */
HmlErrCode hmlGetSecs(double *cpuSecs, double *wallSecs) {
  HML_ERR_PROLOGUE;
#ifdef WIN32
  LARGE_INTEGER time, freq;

  if (cpuSecs)
    *cpuSecs = (double)clock() / CLOCKS_PER_SEC;
  if (wallSecs) {
    if (!QueryPerformanceFrequency(&freq)){
      *wallSecs = 0.0; /* Handle error */
      return;
    }
    if (!QueryPerformanceCounter(&time)){
      *wallSecs = 0.0; /* Handle error */
      return;
    }
    *wallSecs = (double)time.QuadPart / freq.QuadPart;
  }
#else
  struct tms time;
  struct timeval tm;

  if (cpuSecs) {
    times( &time );
    *cpuSecs = (double)( time.tms_utime + time.tms_stime ) / (double)sysconf(_SC_CLK_TCK);
  }
  if (wallSecs) {
    gettimeofday( &tm, NULL );
    *wallSecs = (double)tm.tv_sec + (double)tm.tv_usec/1000000.0;
  }
#endif
  HML_NORMAL_RETURN;
}

HmlErrCode hmlGetTime(HmlTime *time) {
  return hmlGetSecs(&time->cpuTime, &time->wallTime);
}

/* return a randomly generated float */
float
hmlRandomFloat()
{
  return ((float)rand()) / RAND_MAX;
}

FILE*
openFile(const char *filename, const char *mode)
{
  FILE *file = fopen(filename, mode);
  if (!file) {
    fprintf(stderr, "; Error: Cannot open file '%s'\n", filename);
    exit(EXIT_FAILURE);
  }
  return file;
}

void
hmlHandleError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "; Error: '%s' in %s at line %d\n", cudaGetErrorString( err ),
            file, line );
    cudaDeviceReset();
    exit( EXIT_FAILURE );
  }
}

/* see the following URL:
 * https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
 */
int
hmlMaxNumRegistersPerThread(const cudaDeviceProp *prop)
{
  if (prop->major == 1)
    return 124;
  else if (prop->major == 2)
    return 63;
  else if (prop->major == 3) {
    if (prop->minor == 0)
      return 63;
    else
      return 255;
  }
  return 255;
}

uint32_t *
hmlDeviceUint32ArrayAlloc(int numElems)
{
  uint32_t *devArr;       /* array on CUDA device */
  size_t size = numElems * sizeof(uint32_t);  /* byte size of the array */
  /* allocate array in device memory */
  cudaError_t err = cudaMalloc(&devArr, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "; Error: CUDA malloc '%s'\n",cudaGetErrorString(err));
    exit(1);
  }
  return devArr;
}

uint32_t*
hmlDeviceUint32ArrayAllocBind(int                numElems,
                              texture<uint32_t, 1> &texUint32Arr)
{
  uint32_t *devArr;                          /* array on CUDA device */
  size_t size = numElems * sizeof(uint32_t); /* byte size of A */
  size_t offset = 0;                      /* needed by cudaBindTexture */

  devArr = hmlDeviceUint32ArrayAlloc(numElems);
  /* bind A to texture memory */
  cudaError_t err = cudaBindTexture(&offset, texUint32Arr, devArr, size);
  /* check if offset is non-zero */
  if (offset != 0) {
    /* currently the kernel assumes offset to always be 0,
     * although it shouldn't be difficult to write another
     * kernel that takes in offset as an extra argument to
     * account for such a case
     */
    fprintf(stderr, "; Error: offset (= %lu) != 0\n", offset);
    exit(1);
  }
  if (err != cudaSuccess) {
    fprintf(stderr, "; Error: CUDA bind texture '%s'\n",cudaGetErrorString(err));
    exit(1);
  }
  return devArr;
}

float *
hmlDeviceFloatArrayAlloc(int numElems)
{
  float *devArr;       /* array on CUDA device */
  size_t size = numElems * sizeof(float);  /* byte size of the array */
  /* allocate array in device memory */
  cudaError_t err = cudaMalloc(&devArr, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "; Error: CUDA malloc '%s'\n",cudaGetErrorString(err));
    exit(1);
  }
  return devArr;
}

float*
hmlDeviceFloatArrayAllocLoad(const float       *hostArr,
                             int                numElems)
{
  float *devArr;                          /* array on CUDA device */
  size_t size = numElems * sizeof(float); /* byte size of A */

  devArr = hmlDeviceFloatArrayAlloc(numElems);
  /* copy hostArr from host to device */
  cudaError_t err = cudaMemcpy(devArr, hostArr, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "; Error: CUDA memory copy '%s'\n",cudaGetErrorString(err));
    exit(1);
  }
  return devArr;
}

float*
hmlDeviceFloatArrayAllocBind(int                numElems,
                          texture<float, 1> &texFloatArr)
{
  float *devArr;                          /* array on CUDA device */
  size_t size = numElems * sizeof(float); /* byte size of A */
  size_t offset = 0;                      /* needed by cudaBindTexture */

  devArr = hmlDeviceFloatArrayAlloc(numElems);
  /* bind A to texture memory */
  cudaError_t err = cudaBindTexture(&offset, texFloatArr, devArr, size);
  /* check if offset is non-zero */
  if (offset != 0) {
    /* currently the kernel assumes offset to always be 0,
     * although it shouldn't be difficult to write another
     * kernel that takes in offset as an extra argument to
     * account for such a case
     */
    fprintf(stderr, "; Error: offset (= %lu) != 0\n", offset);
    exit(1);
  }
  if (err != cudaSuccess) {
    fprintf(stderr, "; Error: CUDA bind texture '%s'\n",cudaGetErrorString(err));
    exit(1);
  }
  return devArr;
}

float*
hmlDeviceFloatArrayAllocLoadBind(const float       *hostArr,
                              int                numElems,
                              texture<float, 1> &texFloatArr)
{
  float *devArr;                          /* array on CUDA device */
  size_t size = numElems * sizeof(float); /* byte size of A */
  size_t offset = 0;                      /* needed by cudaBindTexture */

  devArr = hmlDeviceFloatArrayAlloc(numElems);
  /* copy hostArr from host to device */
  cudaMemcpy(devArr, hostArr, size, cudaMemcpyHostToDevice);
  /* bind A to texture memory */
  cudaError_t err = cudaBindTexture(&offset, texFloatArr, devArr, size);
  /* check if offset is non-zero */
  if (offset != 0) {
    /* currently the kernel assumes offset to always be 0,
     * although it shouldn't be difficult to write another
     * kernel that takes in offset as an extra argument to
     * account for such a case
     */
    fprintf(stderr, "; Error: offset (= %lu) != 0\n", offset);
    exit(1);
  }
  if (err != cudaSuccess) {
    fprintf(stderr, "; Error: CUDA bind texture '%s'\n",cudaGetErrorString(err));
    exit(1);
  }
  return devArr;
}

void
hmlDevicePropertyCheck(const cudaDeviceProp *prop)
{
  bool pass = true;

  if (prop->warpSize != cHmlThreadsPerWarp) {
    fprintf(stderr, "; Error: Warp size (%d) != %d\n", prop->warpSize, cHmlThreadsPerWarp);
    pass = false;
  }
  if (prop->maxThreadsPerBlock != cHmlMaxThreadsPerBlock) {
    fprintf(stderr, "; Error: Max threads per block (%d) != %d\n",
            prop->maxThreadsPerBlock, cHmlMaxThreadsPerBlock);
    pass = false;
  }
  if (prop->sharedMemPerBlock != cHmlMaxSmemBytes) {
    fprintf(stderr, "; Error: Max shared mem per block (%ld) != %d\n",
            prop->sharedMemPerBlock, cHmlMaxSmemBytes);
    pass = false;
  }
  if (prop->totalConstMem != cHmlMaxCmemBytes) {
    fprintf(stderr, "; Error: Max constant mem (%ld) != %d\n",
            prop->totalConstMem, cHmlMaxCmemBytes);
    pass = false;
  }
  if (prop->maxThreadsDim[0] < cHmlMaxBlockDimX ||
      prop->maxThreadsDim[1] < cHmlMaxBlockDimY ||
      prop->maxThreadsDim[2] < cHmlMaxBlockDimZ) {
    fprintf(stderr, "; Error: Max threads dimensions (%d, %d, %d) < (%d, %d, %d)\n",
            prop->maxThreadsDim[0],
            prop->maxThreadsDim[1],
            prop->maxThreadsDim[2],
            cHmlMaxBlockDimX,
            cHmlMaxBlockDimY,
            cHmlMaxBlockDimZ);
    pass = false;
  }
  if (prop->maxGridSize[0] < cHmlMaxGridDimX ||
      prop->maxGridSize[1] < cHmlMaxGridDimY ||
      prop->maxGridSize[2] < cHmlMaxGridDimZ) {
    fprintf(stderr, "; Error: Max grid dimensions (%d, %d, %d) < (%d, %d, %d)\n",
            prop->maxGridSize[0],
            prop->maxGridSize[1],
            prop->maxGridSize[2],
            cHmlMaxGridDimX,
            cHmlMaxGridDimY,
            cHmlMaxGridDimZ);
    pass = false;
  }
  if (prop->maxTexture1DLinear < cHmlMaxCudaTexture1DLinear) {
    fprintf(stderr, "; Error: Max texture 1D linear = %d < %d\n",
            prop->maxTexture1DLinear, cHmlMaxCudaTexture1DLinear);
    pass = false;
  }
  if (!pass) {
    fprintf(stderr, "; Error: Device property checking failed\n");
    exit(EXIT_FAILURE);
  }
}

void
hmlDevicePropertyPrint(const cudaDeviceProp *prop)
{
  fprintf(stderr, "; Info:   --- General CUDA Information ---\n");
  fprintf(stderr, "cuda_device_name=\"%s\"\n", prop->name);
  fprintf(stderr, "cuda_compute_capability=%d.%d\n", prop->major, prop->minor);
  fprintf(stderr, "cuda_clock_rate=%d\n", prop->clockRate);
  fprintf(stderr, "cuda_device_copy_overlap=");
  if (prop->deviceOverlap)
    fprintf(stderr, "1\n");
  else
    fprintf(stderr, "0\n");
  fprintf(stderr, "cuda_kernel_execution_timeout=");
  if (prop->kernelExecTimeoutEnabled)
    fprintf(stderr, "1\n");
  else
    fprintf(stderr, "0\n");
  fprintf(stderr, "; Info:   --- CUDA Memory Information ---\n");
  fprintf(stderr, "cuda_total_global_mem=%ld\n", prop->totalGlobalMem);
  fprintf(stderr, "cuda_total_constant_mem=%ld\n", prop->totalConstMem);
  fprintf(stderr, "cuda_max_mem_pitch=%ld\n", prop->memPitch);
  fprintf(stderr, "cuda_texture_alignment=%ld\n", prop->textureAlignment);

  fprintf(stderr, "; Info   --- CUDA MP Information ---\n");
  fprintf(stderr, "cuda_multiprocessor_count=%d\n",
          prop->multiProcessorCount);
  fprintf(stderr, "cuda_shared_mem_per_mp=%ld\n", prop->sharedMemPerBlock);
  fprintf(stderr, "cuda_registers_per_mp=%d\n", prop->regsPerBlock);
  fprintf(stderr, "cuda_threads_in_warp=%d\n", prop->warpSize);
  fprintf(stderr, "cuda_max_threads_per_block=%d\n",
          prop->maxThreadsPerBlock);
  fprintf(stderr, "cuda_max_thread_dimension_x=%d\n", prop->maxThreadsDim[0]);
  fprintf(stderr, "cuda_max_thread_dimension_y=%d\n", prop->maxThreadsDim[1]);
  fprintf(stderr, "cuda_max_thread_dimension_z=%d\n", prop->maxThreadsDim[2]);
  fprintf(stderr, "cuda_max_grid_dimension_x=%d\n", prop->maxGridSize[0]);
  fprintf(stderr, "cuda_max_grid_dimension_y=%d\n", prop->maxGridSize[1]);
  fprintf(stderr, "cuda_max_grid_dimension_z=%d\n", prop->maxGridSize[2]);
  fprintf(stderr, "\n");
  fprintf(stderr, "; Info:   --- CUDA Texture Information ---\n");
  fprintf(stderr, "cuda_max_texture_1d=%d\n", prop->maxTexture1D);
  fprintf(stderr, "cuda_max_texture_1d_linear=%d\n", prop->maxTexture1DLinear);
  fprintf(stderr, "\n");
}
