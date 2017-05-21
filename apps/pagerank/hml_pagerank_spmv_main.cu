/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include <stdio.h>
#ifdef WIN32
#include "getopt_win32.h"
#else
#include <unistd.h>
#endif
#include <vector>
#include "hml.h"

using namespace std;

#define cHmlPagerankDampingFactorDefault   0.85f
#define cHmlPagerankOutFileNameExtension   "k2v"

int main(int argc, char *argv[])
{
  char                       helpMsg[] =
    "This program does GPU-based PageRank.\n"
    "Options:\n"
    "\t-c Cross check with CPU-based PageRank results\n"
    "\t-d <damping factor>\n"
    "\t-h Print this help message\n"
    "\t-i <number of PageRank iteration>\n"
    "\t-k <number of top-ranked pages to be printed>\n"
    "\t-o <output file name prefix>\n"
    "\t-g <graph file name>\n"
    "\t-u <GPU ID in [0, #GPUs - 1]>\n"
    "\t-y <verbosity level in [0,2]>\n";
  int             option;
  uint32_t          verbosity = 0;
  HmlGraph        hostGraphVal;
  HmlGraph       *hostGraph = &hostGraphVal;
  vector<float> dampingFactors;
  float         dampingFactor = 0.85;
  uint32_t          numIters = 10;
  uint32_t          printTopK = (uint32_t)-1;
  int             gpuId = 0;
  int             count;
  cudaDeviceProp  prop;
  char           *graphFileName = NULL;
  char           *outFilenamePrefix = NULL;
  double          cpuStart;
  double          cpuEnd;
  double          wallStart;
  double          wallEnd;
  bool            runCPU = false;
  size_t          freeBytesStart;
  size_t          totalBytesStart;
  size_t          freeBytesEnd;
  size_t          totalBytesEnd;
  
  /* get program options */
  while ((option = getopt(argc, argv, ":cd:hi:k:o:r:u:y:")) != -1) {
    switch (option) {
    case 'c':
      runCPU = true;
      break;
      
    case 'd':
      dampingFactor = atof(optarg);
      dampingFactors.push_back(dampingFactor);
      if (verbosity >= 1)
        fprintf(stderr, "; Info: damping factor = %lf\n", dampingFactor);
      break;

    case 'g':
      graphFileName = optarg;
      break;

    case 'h':
      fprintf(stderr, "Help:\n%s\n", helpMsg);
      exit(EXIT_FAILURE);
      break;
  
    case 'i':
      numIters = atoi(optarg);
      break;
  
    case 'k':
      printTopK = atoi(optarg);
      break;
  
    case 'o':
      outFilenamePrefix = optarg;
      break;

    case 'u':
      gpuId = atoi(optarg);
      break;

    case 'y':
      verbosity = atoi(optarg);
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
    graphFileName = argv[optind];
  }

  HANDLE_ERROR(cudaGetDeviceCount(&count));
  if (verbosity >= 2) {
    for (int i = 0; i < count; ++i) {
      HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
      fprintf(stderr, "[cuda_device_%d]\n", i);
      hmlDevicePropertyPrint(&prop);
    }
  }
   /* choose which device to run the kernel code */
  if (gpuId >= count) {
    fprintf(stderr, 
      "; Error: Invalid GPU card #%d, resetting to default (0)\n", gpuId);
    gpuId = 0;
  }
  HANDLE_ERROR(cudaSetDevice(gpuId));
  if (verbosity >= 2) {
    fprintf(stderr, "; Info: Set cuda device to %d\n", gpuId);
    HANDLE_ERROR(cudaMemGetInfo(&freeBytesStart, &totalBytesStart));
    fprintf(stderr, 
            "; Info: Free memory = %ld bytes, total memory = %ld bytes\n",
            freeBytesStart, totalBytesStart);
  }
  if (verbosity >= 1) 
    hmlGetSecs(&cpuStart, &wallStart);
  hmlGraphReadTsv4(graphFileName, false, hostGraph);
  if (verbosity >= 1) {
    hmlGetSecs(&cpuEnd, &wallEnd);
    fprintf(stderr, 
            "; Info: HmlGraph reading: cpu time = %.2lf, wall time = %.2lf\n",
            (cpuEnd - cpuStart) * 1000, (wallEnd - wallStart) * 1000);
  }
  if (verbosity >= 2)
    hmlGraphPrintStats(stderr, hostGraph);
  /* hmlGraphPrintEdges(stdout, hostGraph, false); */

  if (dampingFactors.empty())
    dampingFactors.push_back(cHmlPagerankDampingFactorDefault);

  /* perform pagerank on CPU ? */
  for (size_t i = 0; i < dampingFactors.size(); ++i) {
    if (runCPU == true) {
      hmlPagerankSpmvCpu(hostGraph, dampingFactors[i], numIters, printTopK,
        outFilenamePrefix, cHmlPagerankOutFileNameExtension);
    }
    else {
      hmlPagerankSpmvGpu(hostGraph, dampingFactors[i], numIters, printTopK,
        outFilenamePrefix, cHmlPagerankOutFileNameExtension, verbosity);
    }
  }
  hmlGraphDeleteHost(hostGraph);
  if (verbosity >= 2) {
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
