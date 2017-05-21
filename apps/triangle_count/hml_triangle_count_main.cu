#include <stdio.h>
#ifdef WIN32
#include "getopt_win32.h"
#else
#include <unistd.h>
#endif
#include <vector>
#include <sstream>
#include "hml.h"

using namespace std;

typedef struct {
  bool    runCpu;
  int32_t   gpuId;
  char   *graphFileName;
} HmlTriangleCountOptions;

static HmlErrCode
hmlTriangleCountOptionsInit(HmlTriangleCountOptions *options) {
  HML_ERR_PROLOGUE;
  memset(options, 0, sizeof(HmlTriangleCountOptions));
  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlTriangleCountOptionsDelete(HmlTriangleCountOptions *options) {
  HML_ERR_PROLOGUE;
  memset(options, 0, sizeof(HmlTriangleCountOptions));
  HML_NORMAL_RETURN;
}

int main(int argc, char *argv[])
{
  char                       helpMsg[] =
    "This program does GPU-based TriangleCount.\n"
    "Options:\n"
    "\t-c Cross check with CPU-based TriangleCount results\n"
    "\t-h Print this help message\n"
    "\t-g <graph file name>\n"
    "\t-u <GPU ID in [0, #GPUs - 1]>\n"
    "\t-y <verbosity level in [0,2]>\n";
  HmlTriangleCount     triangleCount;
  int             option;
  HmlTriangleCountOptions options;
  int             numGpus;
  cudaDeviceProp  prop;
  HmlTime         startTime;
  HmlTime         endTime;
  size_t          freeBytesStart;
  size_t          totalBytesStart;
  size_t          freeBytesEnd;
  size_t          totalBytesEnd;

  hmlTriangleCountInit(&triangleCount);
  hmlTriangleCountOptionsInit(&options);
  /* get program options */
  while ((option = getopt(argc, argv, ":chu:y:")) != -1) {
    switch (option) {
    case 'c':
      options.runCpu = true;
      break;
      
    case 'h':
      fprintf(stderr, "Help:\n%s\n", helpMsg);
      exit(EXIT_FAILURE);
      break;
  
    case 'u':
      options.gpuId = atoi(optarg);
      break;

    case 'y':
      triangleCount.verbosity = atoi(optarg);
      break;

    case ':':
      fprintf(stderr, "; Error: Option -%c requires an argument\n", optopt);
      exit(EXIT_FAILURE);
      break;

    case '?':
      fprintf(stderr, "; Error: Unknown option character '%c'.\n", optopt);
      exit(EXIT_FAILURE);
    }
  }
  /* the last argument is the name of the graph file, which
   * is not optional
   */
  if (optind == argc - 1) {
    options.graphFileName = argv[optind];
  }

  HANDLE_ERROR(cudaGetDeviceCount(&numGpus));
  if (triangleCount.verbosity >= 2) {
    for (int i = 0; i < numGpus; ++i) {
      HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
      fprintf(stderr, "[cuda_device_%d]\n", i);
      hmlDevicePropertyPrint(&prop);
    }
  }
   /* choose which device to run the kernel code */
  if (options.gpuId >= numGpus) {
    fprintf(stderr, 
      "; Error: Invalid GPU card #%d, resetting to default (0)\n", options.gpuId);
    options.gpuId = 0;
  }
  HANDLE_ERROR(cudaSetDevice(options.gpuId));
  if (triangleCount.verbosity >= 2) {
    fprintf(stderr, "; Info: Set cuda device to %d\n", options.gpuId);
    HANDLE_ERROR(cudaMemGetInfo(&freeBytesStart, &totalBytesStart));
    fprintf(stderr, 
            "; Info: Free memory = %ld bytes, total memory = %ld bytes\n",
            freeBytesStart, totalBytesStart);
  }
  hmlGetTime(&startTime);
  /* read the input .tsv2 file while transposing the edges on the fly */
  hmlTriangleCountReadOrderedTsv2FileByName(&triangleCount.cpu,
					    options.graphFileName, true);
  hmlGetTime(&endTime);
  if(triangleCount.verbosity > 0) {
    hmlGraphCorePrintStats(&triangleCount.cpu.core, stderr);
    fprintf(stderr, "; Progress: Finished reading input .tsv2 file\n");
    fprintf(stderr, "; Time: %.3f\n", endTime.wallTime - startTime.wallTime);
    fflush(stderr);
  }
  if (!options.runCpu) {
    hmlTriangleCountGpuInit(&triangleCount);
  }
  hmlGetTime(&startTime);
  /* perform triangleCount on CPU ? */
  if (!options.runCpu) {
    hmlTriangleCountGpu(&triangleCount);
    fprintf(stdout, "%lu\n", triangleCount.gpu.numTriangles);
  }
  else {
    hmlTriangleCountRun(&triangleCount.cpu);
    fprintf(stdout, "%lu\n", triangleCount.cpu.numTriangles);
  }
  hmlGetTime(&endTime);
  if(triangleCount.verbosity > 0) {
    fprintf(stderr, "; Progress: TriangleCount completed\n");
    fprintf(stderr, "; Time: %.3f\n", endTime.wallTime - startTime.wallTime);
    fflush(stderr);
  }
  hmlTriangleCountDelete(&triangleCount);
  hmlTriangleCountOptionsDelete(&options);
  if (triangleCount.verbosity >= 2) {
    HANDLE_ERROR(cudaMemGetInfo(&freeBytesEnd, &totalBytesEnd));
    fprintf(stderr,
            "; Info: Free memory = %ld bytes, total memory = %ld bytes\n",
            freeBytesEnd, totalBytesEnd);
    if (freeBytesStart != freeBytesEnd || totalBytesStart != totalBytesEnd)
      fprintf(stderr,
              "; Info: Memory leak: %ld bytes, %ld total bytes\n",
              freeBytesStart - freeBytesEnd, totalBytesStart - totalBytesEnd);
  }
  HANDLE_ERROR(cudaDeviceReset());
}
