/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include "hml_kmeans_kernel.h"
#include "hml_kmeans_kernel_init.h"

void
hmlKmeansPrintKernelConfig(FILE *file, const HmlKmeansKernelConfig &config)
{
  HmlKmeansKernelConfigConstDim::const_iterator it;

  for (size_t d = 0; d < config.size(); ++d) {
    if (!config[d].empty()) {
      for (it = config[d].begin(); it != config[d].end(); ++it) {
        fprintf(file, "%ld %d : %d %d %d %d\n", d, it->first,
          it->second.assignUseTextureMem, it->second.assignOpt,
          it->second.updateUseTextureMem, it->second.updateOpt);
      }
    }
  }
}

void
hmlKmeansReadKernelConfigFile(const char *fileName, HmlKmeansKernelConfig &config)
{
  FILE       *file;
  char        line[cHmlLineBufferSize];
  char       *str;
  uint32_t      numDims;
  uint32_t      numClusts;
  HmlKmeansKernelOption opt;
  uint32_t      assignOptInt;
  uint32_t      updateOptInt;

  file = fopen(fileName, "rb");
  if (!file) {
    fprintf(stderr, "Cannot open file: %s\n", fileName);
    exit( EXIT_FAILURE );
  }
  while (!feof(file)) {
    while (true) {
      str = fgets(line, cHmlLineBufferSize, file);
      if (!str || *str != '#')
        break;
    }
    if (!str) break;
    sscanf(str, "%d %d : %d %d %d %d\n", &numDims, &numClusts,
           (int *)&opt.assignUseTextureMem, &assignOptInt,
           (int *)&opt.updateUseTextureMem, &updateOptInt);
    opt.assignOpt = (HmlKmeansAssignKernelOption)assignOptInt;
    opt.updateOpt = (HmlKmeansUpdateKernelOption)updateOptInt;
    if (config.size() < numDims + 1)
      config.resize(numDims + 1);
    config[numDims][numClusts] = opt;
  }
  fclose(file);
  //hmlKmeansPrintKernelConfig(stderr, config);
}

/* invariant #1: gridDim.x == numClusts
 * invariant #2: blockDim.x == cHmlThreadsPerWarp
 * Note: this version limits the max 'numClusts' to cHmlMaxGridDimX
 */
__global__ void
hmlKmeansFinalUpdate(float *pCtrds,   /* numDims x numClusts x numBlocks */
                  uint32_t  *pSizes,   /* numClusts x numBlocks */
                  uint32_t   numDims,
                  uint32_t   numClusts)
{
  const uint32_t    tid = threadIdx.x;
  const uint32_t    kid = blockIdx.x;   /* cluster id assigned to the block */
  uint32_t    dim;
  uint32_t    idx;
  uint32_t    blk;
  extern __shared__ float mean[];  /* numDims */
  __shared__ uint32_t  size[cHmlThreadsPerWarp];

  size[tid] = 0;
  blk = tid;
  while (blk < cHmlKmeansUpdateBlocks) {
    idx = kid + numClusts * blk;
    size[tid] += pSizes[idx];
    blk += blockDim.x;
  }
  //__syncthreads();
  for (uint32_t numReducers = blockDim.x / 2; numReducers > 0; numReducers /= 2) {
    if (tid < numReducers)
      size[tid] += size[tid + numReducers];
    //__syncthreads();
  }
  /* write final result to the first update block */
  if (tid == 0)
    pSizes[kid] = size[0];
  //__syncthreads();
  /* sum along each dim for all points in the same cluster */
  dim = tid;
  while (dim < numDims) {
    mean[dim] = 0.0;
    idx = dim + numDims * kid;
    for (uint32_t b = 0; b < cHmlKmeansUpdateBlocks; ++b) {
      mean[dim] += pCtrds[idx];
      idx += numDims * numClusts;
    }
    /* write final result to the first update block */
    pCtrds[dim + numDims * kid] = mean[dim] / size[0];
    dim += blockDim.x; /* == cHmlThreadsPerWarp */
  }
}

void
calcGridDim(dim3  *pGrid,
            uint32_t numBlocks,
            uint32_t gridDimXMin,
            uint32_t gridDimXMax)
{
  uint32_t gridDimX;
  uint32_t gridDimY;

  if (gridDimXMax > cHmlMaxGridDimX) {
    fprintf(stderr, "Err: gridDimXMax = %d > cHmlMaxGridDimX = %d\n",
            gridDimXMax, cHmlMaxGridDimX);
    exit(EXIT_FAILURE);
  }
  /* double gridDimX until gridDimY is no more than cHmlMaxGridDimY */
  for (gridDimX = gridDimXMin; gridDimX <= gridDimXMax; gridDimX *= 2) {
    gridDimY = (numBlocks + gridDimX - 1) / gridDimX;
    if (gridDimY <= cHmlMaxGridDimY)
      break;
  }
  if (gridDimX > gridDimXMax) {
    fprintf(stderr, "Err: gridDimX > gridDimXMax\n");
    exit(EXIT_FAILURE);
  }
  pGrid->x = gridDimX;
  pGrid->y = gridDimY;
  pGrid->z = 1;         /* always = 1 */
}

/* invariant #1: gridDim.x == numClusts &&
 * invariant #2: blockDim.x == cHmlThreadsPerWarp
 */
__global__ void
kMeansResidualKernel(float       *pResidual,  /* numClusts */
                     const float *pCtrds,     /* numDims x numClusts */
                     const float *pCtrdsPrev, /* numDims x numClusts */
                     uint32_t         numDims,
                     uint32_t         numClusts)
{
  const uint32_t  tid = threadIdx.x;
  const uint32_t  bid = blockIdx.x;  /* 1 block per cluster */
  uint32_t  dim;
  float delta;
  __shared__ float residual[cHmlThreadsPerWarp];

  residual[tid] = 0.0;
  if (bid < numClusts) {
    pCtrds += numDims * bid;
    pCtrdsPrev += numDims * bid;
    dim = tid;
    while (dim < numDims) {
      delta = fabs(pCtrds[dim] - pCtrdsPrev[dim]);
      //printf("bid = %d, dim = %d: |%f - %f| = %f\n", bid, dim,
      //  pCtrds[dim], pCtrdsPrev[dim], delta);
      residual[tid] = max(delta, residual[tid]);
      dim += blockDim.x; // must == cHmlThreadsPerWarp
    }
  }
  //__syncthreads();
  for (int numReducers = blockDim.x / 2; numReducers > 0; numReducers /= 2) {
    if (tid < numReducers)
      residual[tid] = max(residual[tid], residual[tid + numReducers]);
    //__syncthreads();
  }
  if (tid == 0 && bid < numClusts)
    pResidual[bid] = residual[0];
}

void
hmlKmeansInitKernelRepo(HmlKmeansKernelRepo *repo)
{
  /* init kernels that do not use texture memory */
  hmlKmeansInitKernelAssignGmem(repo->assignGmem);
  hmlKmeansInitKernelAssignSmem(repo->assignSmem);
  hmlKmeansInitKernelAssignSmemUnroll(repo->assignSmemUnroll);
  hmlKmeansInitKernelUpdateGmem(repo->updateGmem);
  hmlKmeansInitKernelUpdateSmem(repo->updateSmem);
  /* init kernel that use texture memory */
  hmlKmeansInitKernelAssignGmemTex(repo->assignGmemTex);
  hmlKmeansInitKernelAssignSmemTex(repo->assignSmemTex);
  hmlKmeansInitKernelAssignSmemUnrollTex(repo->assignSmemUnrollTex);
  hmlKmeansInitKernelUpdateGmemTex(repo->updateGmemTex);
  hmlKmeansInitKernelUpdateSmemTex(repo->updateSmemTex);

}

void
setKmeansAssignGmemArg(HmlKernelArg *kernelArg,
                       uint32_t     numDims,
                       uint32_t     numRows,
                       uint32_t     numClusts)
{
  kernelArg->block.x = kernelArg->block.y = 16;
  kernelArg->block.z = 1;
  uint32_t numThreadsPerBlock = kernelArg->block.x * kernelArg->block.y;
  uint32_t numBlocks = (numRows + numThreadsPerBlock - 1) / numThreadsPerBlock;
  calcGridDim(&kernelArg->grid, numBlocks, 1, cHmlMaxGridDimX);
  kernelArg->allocBytes = 0;
}

void
setKmeansAssignSmemArg(HmlKernelArg *kernelArg,
                       uint32_t     numDims,
                       uint32_t     numClusts)
{
  kernelArg->block.x = 256; /* better than 32x32 block size */
  kernelArg->block.y = 1;
  kernelArg->block.z = 1;
  kernelArg->grid.x = 1024; /* better than 512 */
  kernelArg->grid.y = kernelArg->grid.z = 1;
  kernelArg->allocBytes = sizeof(float) * numDims * numClusts;
}

void
setKmeansAssignSmemUnrollArg(HmlKernelArg *kernelArg,
                             uint32_t     numDims,
                             uint32_t     numClusts)
{
  kernelArg->block.x = 256; /* better than 32x32 block size */
  kernelArg->block.y = 1;
  kernelArg->block.z = 1;
  kernelArg->grid.x = 1024; /* better than 512 */
  kernelArg->grid.y = kernelArg->grid.z = 1;
  /* round up numClusts to the nearest multiple of 16, which is required
  * by hmlKmeansAssignCtrdSmemUnroll kernels */
  uint32_t numClusts16 = (numClusts + 15) / 16 * 16;
  kernelArg->allocBytes = sizeof(float) * numDims * numClusts16;
}

void
hmlKmeansSetUpdateGmemArg(HmlKernelArg *kernelArg)
{
  kernelArg->block.x = cHmlThreadsPerWarp;
  kernelArg->block.y = kernelArg->block.z = 1;
  kernelArg->grid.x = cHmlKmeansUpdateBlocks;
  kernelArg->grid.y = kernelArg->grid.z = 1;
  kernelArg->allocBytes = 0;
}

void
hmlKmeansSetUpdateSmemArg(HmlKernelArg *kernelArg,
                       uint32_t     numDims,
                       uint32_t     numClusts)
{
  kernelArg->block.x = cHmlThreadsPerWarp;
  kernelArg->block.y = kernelArg->block.z = 1;
  kernelArg->grid.x = cHmlKmeansUpdateBlocks;
  kernelArg->grid.y = kernelArg->grid.z = 1;
  kernelArg->allocBytes = sizeof(uint32_t) * numClusts +
    sizeof(float) * numDims * numClusts;
}

void
kmeansSelectKernel(HmlKmeansAssignKernel       *assignKernel,
                   cudaFuncCache            *assignCacheConfig,
                   HmlKernelArg                *assignArg,
                   HmlKmeansUpdateKernel       *updateKernel,
                   cudaFuncCache            *updateCacheConfig,
                   HmlKernelArg                *updateArg,
                   const HmlKmeansKernelRepo   *repo,
                   const HmlKmeansKernelConfig &config,
                   uint32_t                    numDims,
                   uint32_t                    numRows,
                   uint32_t                    numClusts)
{
  HmlKmeansKernelConfigConstDim::const_iterator it;

  *assignKernel = NULL;
  *updateKernel = NULL;
  if (config[numDims].empty()) {
    fprintf(stderr, "Error: missing kernel configuration for #dims = %d\n",
      numDims);
    exit(EXIT_FAILURE);
  }
  for (it = config[numDims].begin(); it != config[numDims].end(); ++it) {
    if (it->first == numClusts) {
      switch (it->second.assignOpt) {
      case eHmlKmeansAssignGlobalMem:
        *assignKernel = (it->second.assignUseTextureMem) ?
          repo->assignGmemTex[numDims] : repo->assignGmem[numDims];
        *assignCacheConfig = cudaFuncCachePreferL1;
        setKmeansAssignGmemArg(assignArg, numDims, numRows, numClusts);
        //fprintf(stderr, "Info: eHmlKmeansAssignGlobalMem\n");
        break;
      case eHmlKmeansAssignSharedMem:
        *assignKernel = (it->second.assignUseTextureMem) ?
          repo->assignSmemTex[numDims] : repo->assignSmem[numDims];
        *assignCacheConfig = cudaFuncCachePreferShared;
        setKmeansAssignSmemArg(assignArg, numDims, numClusts);
        //fprintf(stderr, "Info: eHmlKmeansAssignSharedMem\n");
        break;
      case eHmlKmeansAssignSharedMemUnroll:
        *assignKernel = (it->second.assignUseTextureMem) ?
          repo->assignSmemUnrollTex[numDims] : repo->assignSmemUnroll[numDims];
        *assignCacheConfig = cudaFuncCachePreferShared;
        setKmeansAssignSmemUnrollArg(assignArg, numDims, numClusts);
        //fprintf(stderr, "Info: eHmlKmeansAssignSharedMemUnroll\n");
        break;
      }
      switch (it->second.updateOpt) {
      case eHmlKmeansUpdateGlobalMem:
        *updateKernel = (it->second.updateUseTextureMem) ?
          repo->updateGmemTex[numDims] : repo->updateGmem[numDims];
        *updateCacheConfig = cudaFuncCachePreferL1;
        hmlKmeansSetUpdateGmemArg(updateArg);
        //fprintf(stderr, "Info: eHmlKmeansUpdateGlobalMem\n");
        break;
      case eHmlKmeansUpdateSharedMem:
        *updateKernel = (it->second.updateUseTextureMem) ?
          repo->updateSmemTex[numDims] : repo->updateSmem[numDims];
        *updateCacheConfig = cudaFuncCachePreferShared;
        hmlKmeansSetUpdateSmemArg(updateArg, numDims, numClusts);
        //fprintf(stderr, "Info: eHmlKmeansUpdateSharedMem\n");
        break;
      }
    }
  }
}
