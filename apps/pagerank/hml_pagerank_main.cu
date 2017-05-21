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
#include <sstream>
#include "hml.h"

using namespace std;

#define cHmlPagerankNumItersDefault        10
#define cHmlPagerankOutFileNameExtension   "k2v"

typedef struct {
  bool    runCpu;
  int32_t gpuId;
  vector<float> dampingFactors;
  char   *graphFileName;
  FILE   *graphFile;
  char   *inOutDegreeInputFileName;
  FILE   *inOutDegreeInputFile;
  char   *outFileNamePrefix;
} HmlPagerankOptions;

static HmlErrCode
hmlPagerankOptionsInit(HmlPagerankOptions *options) {
  HML_ERR_PROLOGUE;
  memset(options, 0, sizeof(HmlPagerankOptions));
  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlPagerankOptionsDelete(HmlPagerankOptions *options) {
  HML_ERR_PROLOGUE;
  memset(options, 0, sizeof(HmlPagerankOptions));
  HML_NORMAL_RETURN;
}

static HmlErrCode
hmlPagerankOptionsOpenFiles(HmlPagerankOptions *options) {
  HML_ERR_PROLOGUE;

  if(options->graphFileName) {
    options->graphFile = fopen(options->graphFileName, "rb");
    HML_ERR_GEN(!options->graphFile, cHmlErrGeneral);
  }
  else {
    options->graphFile = stdin;
  }
  if(options->inOutDegreeInputFileName) {
    options->inOutDegreeInputFile = fopen(options->inOutDegreeInputFileName, "rb");
    HML_ERR_GEN(!options->inOutDegreeInputFile, cHmlErrGeneral);
  }
  else {
    options->inOutDegreeInputFile = NULL;
  }

  HML_NORMAL_RETURN;
}

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
  HmlPagerank     pagerank;
  int             option;
  HmlPagerankOptions options;
  FILE           *outFile;
  ostringstream   outFilename;
  float         dampingFactor = 0.85;
  int             numGpus;
  cudaDeviceProp  prop;
  HmlTime         startTime;
  HmlTime         endTime;
  size_t          freeBytesStart;
  size_t          totalBytesStart;
  size_t          freeBytesEnd;
  size_t          totalBytesEnd;

  hmlPagerankInit(&pagerank);
  hmlPagerankOptionsInit(&options);
  /* get program options */
  while ((option = getopt(argc, argv, ":cd:D:hi:k:o:u:y:")) != -1) {
    switch (option) {
    case 'c':
      options.runCpu = true;
      break;
      
    case 'd':
      dampingFactor = atof(optarg);
      options.dampingFactors.push_back(dampingFactor);
      if (pagerank.verbosity >= 1)
        fprintf(stderr, "; Info: damping factor = %lf\n", dampingFactor);
      break;

    case 'D':
      options.inOutDegreeInputFileName = optarg;
      break;

    case 'h':
      fprintf(stderr, "Help:\n%s\n", helpMsg);
      exit(EXIT_FAILURE);
      break;
  
    case 'i':
      pagerank.numIters = atoi(optarg);
      break;
  
    case 'k':
      pagerank.topK = atoi(optarg);
      break;
  
    case 'o':
      options.outFileNamePrefix = optarg;
      break;

    case 'u':
      options.gpuId = atoi(optarg);
      break;

    case 'y':
      pagerank.verbosity = atoi(optarg);
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
  if (pagerank.verbosity >= 2) {
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
  if (pagerank.verbosity >= 2) {
    fprintf(stderr, "; Info: Set cuda device to %d\n", options.gpuId);
    HANDLE_ERROR(cudaMemGetInfo(&freeBytesStart, &totalBytesStart));
    fprintf(stderr, 
            "; Info: Free memory = %ld bytes, total memory = %ld bytes\n",
            freeBytesStart, totalBytesStart);
  }
  hmlPagerankOptionsOpenFiles(&options);
  hmlGetTime(&startTime);
  /* gather statistics from either a precomputed degree file, or
   * from the .tsv2 file itself
   */
  hmlPagerankSetInputFiles(&pagerank, options.graphFile,
                           options.inOutDegreeInputFile);
  hmlGetTime(&endTime);
  if(pagerank.verbosity > 0) {
    fprintf(stderr, "; Progress: |Vsrc| = %u, |E| = %lu\n",
            pagerank.maxNumSrcVertices,
            pagerank.numEdges);
    fprintf(stderr, "; Time: %.3f\n", endTime.wallTime - startTime.wallTime);
    fflush(stderr);
  }
  hmlGetTime(&startTime);
  /* read the input .tsv2 file while transposing the edges on the fly */
  hmlPagerankReadTsv2InFile(&pagerank);
  hmlGetTime(&endTime);
  if(pagerank.verbosity > 0) {
    fprintf(stderr, "; Progress: Finished reading input .tsv2 file\n");
    fprintf(stderr, "; Time: %.3f\n", endTime.wallTime - startTime.wallTime);
    fflush(stderr);
  }
  if (options.dampingFactors.empty())
    options.dampingFactors.push_back(cHmlPagerankDampingFactorDefault);

  if (!options.runCpu) {
    hmlPagerankGpuInit(&pagerank);
  }
  /* perform pagerank on CPU ? */
  for (size_t i = 0; i < options.dampingFactors.size(); ++i) {
    pagerank.dampingFactor = options.dampingFactors[i];
    hmlGetTime(&startTime);
    if (!options.runCpu) {
      hmlPagerankGpu(&pagerank);
    }
    else {
      hmlPagerankCpu(&pagerank);
    }
    hmlGetTime(&endTime);
    if(pagerank.verbosity > 0) {
      fprintf(stderr, "; Progress: PageRank completed\n");
      fprintf(stderr, "; Time: %.3f\n", endTime.wallTime - startTime.wallTime);
      fflush(stderr);
    }

    hmlGetTime(&startTime);
    hmlPagerankFindTopK(&pagerank);
    hmlGetTime(&endTime);
    if(pagerank.verbosity > 0) {
      fprintf(stderr, "; Progress: Found top-k pages\n");
      fprintf(stderr, "; Time: %.3f\n", endTime.wallTime - startTime.wallTime);
      fflush(stderr);
    }
    hmlGetTime(&startTime);
    if (options.outFileNamePrefix) {
      outFilename << options.outFileNamePrefix << "(d=" << dampingFactor << ")."
                  << cHmlPagerankOutFileNameExtension;
      outFile = openFile(outFilename.str().c_str(), "wb");
    }
    else {
      outFile = stdout;
    }
    hmlPagerankPrintTopK(&pagerank, outFile);
    if (outFile != stdout)
      fclose(outFile);
    hmlGetTime(&endTime);
    if(pagerank.verbosity > 0) {
      fprintf(stderr, "; Progress: Printed top-k pages\n");
      fprintf(stderr, "; Time: %.3f\n", endTime.wallTime - startTime.wallTime);
      fflush(stderr);
    }
  }
  hmlPagerankDelete(&pagerank);
  hmlPagerankOptionsDelete(&options);
  if (pagerank.verbosity >= 2) {
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
