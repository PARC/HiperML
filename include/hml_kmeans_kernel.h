/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_KMEANS_KERNEL_H_INCLUDED_
#define HML_KMEANS_KERNEL_H_INCLUDED_

#include "hml_common.h"
#include <vector>
#include <map>

#define cHmlKmeansMaxDims      64
#define cHmlKmeansUpdateBlocks 1024

typedef enum {
  eHmlKmeansAssignGlobalMem = 0,
  eHmlKmeansAssignSharedMem,
  eHmlKmeansAssignSharedMemUnroll
} HmlKmeansAssignKernelOption;

typedef enum {
  eHmlKmeansUpdateGlobalMem = 0,
  eHmlKmeansUpdateSharedMem
} HmlKmeansUpdateKernelOption;

typedef struct {
  bool                     assignUseTextureMem;
  HmlKmeansAssignKernelOption assignOpt;
  bool                     updateUseTextureMem;
  HmlKmeansUpdateKernelOption updateOpt;
} HmlKmeansKernelOption;

/* mapping from number of clusters to the best kernel option
 * for a constant dimension
 */
typedef std::map<int, HmlKmeansKernelOption> HmlKmeansKernelConfigConstDim;

typedef std::vector<HmlKmeansKernelConfigConstDim> HmlKmeansKernelConfig;

typedef void (*HmlKmeansAssignKernel)(uint32_t*        pAsmnts,
                                   const float *pRows,
                                   uint32_t         numRows,
                                   const float *pCtrds,
                                   uint32_t         numClusts);

typedef void (*HmlKmeansUpdateKernel)(float       *pCtrds,
                                   uint32_t        *pSizes,
                                   uint32_t         numClusts,
                                   const uint32_t  *pAsmnts,
                                   const float *pRows,
                                   uint32_t         numRows);

typedef struct {
  /* kernels that do not use texture memory */
  HmlKmeansAssignKernel assignGmem[cHmlKmeansMaxDims + 1];
  HmlKmeansAssignKernel assignSmem[cHmlKmeansMaxDims + 1];
  HmlKmeansAssignKernel assignSmemUnroll[cHmlKmeansMaxDims + 1];
  HmlKmeansUpdateKernel updateGmem[cHmlKmeansMaxDims + 1];
  HmlKmeansUpdateKernel updateSmem[cHmlKmeansMaxDims + 1];
  /* kernels that use texture memory */
  HmlKmeansAssignKernel assignGmemTex[cHmlKmeansMaxDims + 1];
  HmlKmeansAssignKernel assignSmemTex[cHmlKmeansMaxDims + 1];
  HmlKmeansAssignKernel assignSmemUnrollTex[cHmlKmeansMaxDims + 1];
  HmlKmeansUpdateKernel updateGmemTex[cHmlKmeansMaxDims + 1];
  HmlKmeansUpdateKernel updateSmemTex[cHmlKmeansMaxDims + 1];
} HmlKmeansKernelRepo;


void hmlKmeansReadInputFile(const char  *fileName,
                         float    **ppData,
                         uint32_t      *pNumColumns,
                         uint32_t      *pNumRows);

void hmlKmeansReadKernelConfigFile(const char *fileName,
                                HmlKmeansKernelConfig &config);

void hmlKmeansPrintKernelConfig(FILE *file, const HmlKmeansKernelConfig &config);

void hmlKmeansInitKernelRepo(HmlKmeansKernelRepo *repo);

void kmeansSelectKernel(HmlKmeansAssignKernel       *assignKernel,
                        cudaFuncCache            *assignCacheConfig,
                        HmlKernelArg                *assignArg,
                        HmlKmeansUpdateKernel       *updateKernel,
                        cudaFuncCache            *updateCacheConfig,
                        HmlKernelArg                *updateArg,
                        const HmlKmeansKernelRepo   *repo,
                        const HmlKmeansKernelConfig &config,
                        uint32_t                    numDims,
                        uint32_t                    numRows,
                        uint32_t                    numClusts);

/* invariant #1: gridDim.x == numClusts
 * invariant #2: blockDim.x == cHmlThreadsPerWarp
 * Note: this version limits the max 'numClusts' to cHmlMaxGridDimX
 */
__global__ void
hmlKmeansFinalUpdate(float *pCtrds,   /* numDims x numClusts x numBlocks */
                  uint32_t  *pSizes,   /* numClusts x numBlocks */
                  uint32_t   numDims,
                  uint32_t   numClusts);

/* invariant #1: gridDim.x == numClusts &&
 * invariant #2: blockDim.x == cHmlThreadsPerWarp
 */
__global__ void
kMeansResidualKernel(float       *pResidual,  /* numClusts */
                     const float *pCtrds,     /* numDims x numClusts */
                     const float *pCtrdsPrev, /* numDims x numClusts */
                     uint32_t         numDims,
                     uint32_t         numClusts);

#endif /* HML_KMEANS_KERNEL_H_INCLUDED_ */
