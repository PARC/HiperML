/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include <stdio.h>
#include <vector>
#include <map>
#include <unistd.h>
#include <float.h>

#include "hml_kmeans.h"

int main(int argc, char *argv[])
{
  int             option;
  double          cpuStart;
  double          cpuEnd;
  double          wallStart;
  double          wallEnd;
  int             gpuId = 0;
  char           *inputFileName = NULL;
  char           *kernelConfigFileName = NULL;
  float        *pRows;
  uint32_t          numIters = 10;
  uint32_t          numClusts = 10;
  uint32_t          numDims;
  uint32_t          numRows;
  int             count;
  cudaDeviceProp  prop;
  bool            runCPU = false;
  size_t          freeBytesStart;
  size_t          totalBytesStart;
  size_t          freeBytesEnd;
  size_t          totalBytesEnd;
  float        *pCtrds;
  uint32_t         *pAsmnts;
  uint32_t         *pSizes;
  double         stopResidual = 0.000001;
  float         residual;
  HmlKmeansKernelConfig kernelConfig;
  HmlKmeansKernelRepo   kernelRepo;
  uint32_t             verbosity = 0;

  /* get program options */
  while ((option = getopt(argc, argv, ":cf:g:i:k:r:s:v:")) != -1) {
    switch (option) {
    case 'c':
      runCPU = true;
      break;

    case 'f':
      inputFileName = optarg;
      break;

    case 'g':
      gpuId = atoi(optarg);
      break;

    case 'i':
      numIters = atoi(optarg);
      break;

    case 'k':
      numClusts = atoi(optarg);
      if (cHmlMaxGridDimX < numClusts) {
        fprintf(stderr, "Error: *** cHmlMaxGridDimX = %d < k = %d ***\n",
          cHmlMaxGridDimX, numClusts);
        fprintf(stderr, "Tip: Incease cHmlMaxGridDimX and recompile with "
                        "'compute_30,sm_30'\n");
        exit(EXIT_FAILURE);
      }
      break;

    case 'r':
      stopResidual = atof(optarg);
      break;

    case 's':
      kernelConfigFileName = optarg;
      break;

    case 'v':
      verbosity = atoi(optarg);
      break;

    case ':':
      fprintf(stderr, "Option -%c requires an argument\n", optopt);
      exit(EXIT_FAILURE);
      break;

    case '?':
      fprintf(stderr, "Unknown option character '%c'.\n", optopt);
      exit(EXIT_FAILURE);
    }
  }
  if (!runCPU && !kernelConfigFileName) {
    fprintf(stderr, "Please specify a kernel config file (-s)\n");
    exit(EXIT_FAILURE);
  }
  HANDLE_ERROR(cudaGetDeviceCount(&count));
  for (int i = 0; i < count; ++i) {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    if (verbosity >= 3) {
      fprintf(stderr, "Device %d:\n", i);
      hmlDevicePropertyPrint(&prop);
    }
    hmlDevicePropertyCheck(&prop);
  }
  /* choose which device to run the kernel code */
  if (gpuId >= count) {
    fprintf(stderr, "Invalid GPU card #%d, resetting to default (0)\n", gpuId);
    gpuId = 0;
  }
  HANDLE_ERROR(cudaSetDevice(gpuId));
  if (verbosity >= 2)
    fprintf(stderr, "Set device to GPU #%d\n", gpuId);

  /* get free and total memory statistics before */
  HANDLE_ERROR(cudaMemGetInfo(&freeBytesStart, &totalBytesStart));
  if (verbosity >= 2)
    fprintf(stderr, "Free memory = %ld bytes, total memory = %ld bytes\n",
            freeBytesStart, totalBytesStart);

  /* read the input file */
  hmlGetSecs(&cpuStart, &wallStart);
  hmlKmeansReadInputFile(inputFileName, &pRows, &numDims, &numRows);
  hmlGetSecs(&cpuEnd, &wallEnd);
  if (verbosity >= 2)
    fprintf(stderr, "Reading input: cpu time = %lf, wall time = %lf\n",
            cpuEnd - cpuStart, wallEnd - wallStart);

  if (numDims * numRows <= 100 && verbosity >= 3) {
    hmlPrintDataMatrix(pRows, numDims, numRows);
  }
  /* round up numClusts to the nearest multiple of 16 */
  uint32_t numClusts16 = (numClusts + 15) / 16 * 16;
  MALLOC(pCtrds, float, numDims * numClusts16);
  /* pad the extra space with FLT_MAX / 2 */
  for (uint32_t k = numClusts; k < numClusts16; ++k) {
    for (uint32_t dim = 0; dim < numDims; ++dim)
      pCtrds[k * numDims + dim] = FLT_MAX / 2;
  }
  MALLOC(pAsmnts, uint32_t, numRows);
  MALLOC(pSizes, uint32_t, numClusts);

  if (runCPU) {
    hmlGetSecs(&cpuStart, &wallStart);
    hmlKmeansCpu(pCtrds, pSizes, pAsmnts, &residual, pRows, numDims,
      numRows, numClusts, numIters, (float)stopResidual);
    hmlGetSecs(&cpuEnd, &wallEnd);
    if (verbosity >= 2) {
      fprintf(stderr, "Residual = %f\n", residual);
      fprintf(stderr, "K-means CPU: cpu time = %lf, wall time = %lf\n",
              cpuEnd - cpuStart, wallEnd - wallStart);
    }
    else if (verbosity == 0)
      fprintf(stderr, "%lf\n", wallEnd - wallStart);
  }
  else {
    hmlKmeansReadKernelConfigFile(kernelConfigFileName, kernelConfig);
    hmlKmeansInitKernelRepo(&kernelRepo);
    hmlGetSecs(&cpuStart, &wallStart);
    hmlKmeansGpu(pCtrds, pSizes, pAsmnts, &residual, pRows, numDims,
              numRows, numClusts, numIters, (float)stopResidual,
              &kernelRepo, kernelConfig, verbosity);
    hmlGetSecs(&cpuEnd, &wallEnd);
    if (verbosity >= 2) {
      fprintf(stderr, "Residual = %f\n", residual);
      fprintf(stderr, "K-means GPU: cpu time = %lf, wall time = %lf\n",
              cpuEnd - cpuStart, wallEnd - wallStart);
    }
  }
  hmlKmeansPrintCluster(pCtrds, pSizes, numDims, numClusts, pAsmnts, numRows);
  FREE(pSizes);
  FREE(pAsmnts);
  FREE(pCtrds);
  FREE(pRows);

  /* get free and total memory statistics after */
  HANDLE_ERROR(cudaMemGetInfo(&freeBytesEnd, &totalBytesEnd));
  if (verbosity >= 2) {
    fprintf(stderr, "Free memory = %ld bytes, total memory = %ld bytes\n",
            freeBytesEnd, totalBytesEnd);

    /* report any potential memory leaks */
    if (freeBytesStart != freeBytesEnd || totalBytesStart != totalBytesEnd)
      fprintf(stderr, "Memory leak: %ld free bytes, %ld total bytes\n",
              freeBytesStart - freeBytesEnd, totalBytesStart - totalBytesEnd);
  }
  HANDLE_ERROR(cudaDeviceReset());
}
