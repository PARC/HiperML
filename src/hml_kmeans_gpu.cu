#include <stdio.h>
/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#include "hml_kmeans_gpu.h"
#include <float.h>
#include <vector>
#include <map>

texture<float, 1> texData1;

/* pRows, pCtrds, pAsmnts, and pSizes must be
 * pre-allocated by the caller.
 */
void
hmlKmeansGpu(float                  *pCtrds,         /* numDims x numClusts */
             uint32_t                   *pSizes,         /* numClusts */
             uint32_t                   *pAsmnts,        /* numRows */
             float                  *pFinalResidual, /* return final residual */
             const float            *pRows,          /* numDims x numRows */
             uint32_t                    numDims,
             uint32_t                    numRows,
             uint32_t                    numClusts,
             uint32_t                    numIters,
             float                   stopResidual,   /* termination residual */
             const HmlKmeansKernelRepo   *repo,
             const HmlKmeansKernelConfig &config,
             uint32_t              verbosity) {
  uint32_t     iter;                /* iterations performed so far */
  uint32_t     numUpdateBlocks = cHmlKmeansUpdateBlocks;
  float   *pDevRows;            /* numDims x numRows */
  float   *pDevMeans;           /* numDims x numClusts x numUpdateBlocks */
  float   *pDevMeansPrev;       /* numDims x numClusts */
  uint32_t    *pDevAssign;          /* numRows */
  uint32_t    *pDevClusterSize;     /* numClusts x numUpdateBlocks */
  float   *pDevResidual;        /* numClusts */
  float   *pResidual;           /* numClusts */
  float    residual = FLT_MAX;
  HmlKmeansAssignKernel assignKernel;
  cudaFuncCache      assignCacheConfig;
  HmlKmeansUpdateKernel updateKernel;
  cudaFuncCache      updateCacheConfig;
  HmlKernelArg  assignArg;
  HmlKernelArg  updateArg;
  size_t     texOffset = 0;
  double     wallStart;
  double     wallEnd;
  double     totalAssignTime = 0.0;
  double     totalUpdateTime = 0.0;
  double     kmeansGpuStartTime;
  double     kmeansGpuEndTime;

  /* check for # of floats (not bytes!) in the dataset,
   * which should not exceed CUDA's maxTexture1DLinear limit
   */
  if(numDims * numRows > cHmlMaxCudaTexture1DLinear) {
    fprintf(stderr, "CUDA maxTexture1DLinear exceeded\n");
    exit(EXIT_FAILURE);
  }

  /* use Forgy method to initialize the cluster means */
  for(uint32_t k = 0; k < numClusts; ++k) {
    /* random selection may pick the same row twice,
     * which will result in empty cluster(s).
     * Thus, we always pick the first k points
     * as the initial centroids, instead of
     * using the following code:
     * uint32_t row = randomUInt32(0, numRows);
     */
    memcpy(pCtrds + k * numDims, pRows + k * numDims,
           sizeof(float) * numDims);
  }

  MALLOC(pResidual, float, numClusts);
  kmeansSelectKernel(&assignKernel, &assignCacheConfig, &assignArg,
                     &updateKernel, &updateCacheConfig, &updateArg,
                     repo, config, numDims, numRows, numClusts);
  if(!assignKernel || !updateKernel) {
    fprintf(stderr, "Error: missing assign or update kernel for #dims = %d\n",
            numDims);
    exit(EXIT_FAILURE);
  }
  hmlGetSecs(NULL, &kmeansGpuStartTime);
  HANDLE_ERROR(cudaFuncSetCacheConfig(assignKernel, assignCacheConfig));
  HANDLE_ERROR(cudaFuncSetCacheConfig(updateKernel, updateCacheConfig));

  /* allocate memory on device */
  HANDLE_ERROR(cudaMalloc((void **)&pDevRows,
                          sizeof(float) * numDims * numRows));
  HANDLE_ERROR(cudaMalloc((void **)&pDevMeans,
                          sizeof(float) * numDims * numClusts *
                          numUpdateBlocks)); /* for parallel update */
  HANDLE_ERROR(cudaMalloc((void **)&pDevMeansPrev,
                          sizeof(float) * numDims * numClusts));
  HANDLE_ERROR(cudaMalloc((void **)&pDevAssign,
                          sizeof(uint32_t) * numRows));
  HANDLE_ERROR(cudaMalloc((void **)&pDevClusterSize,
                          sizeof(uint32_t) * numClusts *
                          numUpdateBlocks)); /* for parallel update */
  HANDLE_ERROR(cudaMalloc((void **)&pDevResidual,
                          sizeof(float) * numClusts));
  /* copy from host to device */
  HANDLE_ERROR(cudaMemcpy((void *)pDevRows, (void *)pRows,
                          sizeof(float) * numDims * numRows,
                          cudaMemcpyHostToDevice));

  /* bind row data to texture */
  HANDLE_ERROR(cudaBindTexture(&texOffset, texData1, pDevRows,
                               sizeof(float) * numDims * numRows));
  /* check for non-zero offset */
  if(texOffset != 0) {
    fprintf(stderr, "Error: Texture offset != 0\n");
    exit(EXIT_FAILURE);
  }

  /* initialize only the first update block, because assign kernel
   * only uses this block; whereas the update kernel uses all
   * 'numUpdateBlocks' blocks during the computation, although only
   * the first one constains the final result
   */
  HANDLE_ERROR(cudaMemcpy((void *)pDevMeans, (void *)pCtrds,
                          sizeof(float) * numDims * numClusts,
                          cudaMemcpyHostToDevice));

  /* perform numIters iterations, unless residual <= stopResidual */
  for(iter = 0; iter < numIters; ++iter) {
    HANDLE_ERROR(cudaMemcpy(pDevMeansPrev, pDevMeans,
                            sizeof(float) * numDims * numClusts,
                            cudaMemcpyDeviceToDevice));
    if(verbosity >= 1) {
      hmlGetSecs(NULL, &wallStart);
    }

    assignKernel<<<assignArg.grid, assignArg.block, assignArg.allocBytes>>>
    (pDevAssign, pDevRows, numRows, pDevMeans, numClusts);

    if(verbosity >= 1) {
      cudaDeviceSynchronize(); //only needed to hmlGetSecs() below
      hmlGetSecs(NULL, &wallEnd);
      totalAssignTime += wallEnd - wallStart;
    }

    /*
    HANDLE_ERROR(cudaMemcpy((void *)pAsmnts, (void *)pDevAssign,
                            sizeof(uint32_t) * numRows,
                            cudaMemcpyDeviceToHost));
    hmlDebugPrintAsmnts(pAsmnts, numRows);
    */

    /* clear memory for pDevMeans */
    HANDLE_ERROR(cudaMemset(pDevMeans, 0,
                            sizeof(float) * numDims * numClusts *
                            numUpdateBlocks)); /* for parallel udpate */
    /* clear memory for pDevClusterSize */
    HANDLE_ERROR(cudaMemset(pDevClusterSize, 0,
                            sizeof(uint32_t) * numClusts *
                            numUpdateBlocks)); /* for parallel update */
    if(verbosity >= 1) {
      hmlGetSecs(NULL, &wallStart);
    }

    updateKernel<<<updateArg.grid, updateArg.block, updateArg.allocBytes>>>
    (pDevMeans, pDevClusterSize, numClusts, pDevAssign, pDevRows, numRows);

    hmlKmeansFinalUpdate<<<numClusts, cHmlThreadsPerWarp, sizeof(float)*numDims>>>
    (pDevMeans,
     pDevClusterSize,
     numDims,
     numClusts);

    if(verbosity >= 1) {
      cudaDeviceSynchronize(); //only needed to hmlGetSecs() below
      hmlGetSecs(NULL, &wallEnd);
      totalUpdateTime += wallEnd - wallStart;
    }

    kMeansResidualKernel<<<numClusts, cHmlThreadsPerWarp>>>(pDevResidual,
        pDevMeans,
        pDevMeansPrev,
        numDims,
        numClusts);

    /*
    HANDLE_ERROR(cudaMemcpy(pCtrds, pDevMeans,
                            sizeof(float) * numDims * numClusts,
                            cudaMemcpyDeviceToHost));
                            */

    /* copy back the residuals, since we need to do the final reduction
     * on CPU
     */
    HANDLE_ERROR(cudaMemcpy(pResidual, pDevResidual,
                            sizeof(float) * numClusts,
                            cudaMemcpyDeviceToHost));
    //hmlGetSecs(NULL, &wallEnd);
    //fprintf(stderr, "GePU non-assign kernels: wall time = %lf\n",
    //        wallEnd - wallStart);

    residual = 0.0;
    for(uint32_t k = 0; k < numClusts; ++k) {
      residual = max(residual, pResidual[k]);
    }
#ifdef _DEBUG
    fprintf(stderr, "Iteration #%d: residual = %f\n", iter + 1, residual);
#endif /* _DEBUG */
    if(residual <= stopResidual) {
      fprintf(stderr, "\nK-means GPU converged at iteration %d\n", iter + 1);
      break;
    }
  }
  *pFinalResidual = residual;
  /* only copy back the first update block */
  HANDLE_ERROR(cudaMemcpy((void *)pCtrds, (void *)pDevMeans,
                          sizeof(float) * numDims * numClusts,
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy((void *)pAsmnts, (void *)pDevAssign,
                          sizeof(uint32_t) * numRows,
                          cudaMemcpyDeviceToHost));
  /* only copy back the first update block */
  HANDLE_ERROR(cudaMemcpy((void *)pSizes, (void *)pDevClusterSize,
                          sizeof(uint32_t) * numClusts,
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaUnbindTexture(texData1));
  HANDLE_ERROR(cudaFree(pDevRows));
  HANDLE_ERROR(cudaFree(pDevMeans));
  HANDLE_ERROR(cudaFree(pDevMeansPrev));
  HANDLE_ERROR(cudaFree(pDevAssign));
  HANDLE_ERROR(cudaFree(pDevClusterSize));
  HANDLE_ERROR(cudaFree(pDevResidual));
  /* restore the default cache config */
  HANDLE_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferNone));

  hmlGetSecs(NULL, &kmeansGpuEndTime);

  FREE(pResidual);

  /* convert wall-clock seconds to milliseconds */
  if(verbosity >= 1) {
    fprintf(stderr, "%10.2lf %10.2lf ", totalAssignTime * 1000.0,
            totalUpdateTime * 1000.0);
  }
  if(verbosity <= 1) {
    fprintf(stderr, "%10.2lf\n", (kmeansGpuEndTime -
                                  kmeansGpuStartTime) * 1000.0);
  }
}
