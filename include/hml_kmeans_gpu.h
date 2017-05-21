/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_KMEANS_GPU_H_INCLUDED_
#define HML_KMEANS_GPU_H_INCLUDED_

#include "hml_common.h"
#include "hml_kmeans_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

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
             uint32_t                    verbosity);

#ifdef __cplusplus
}
#endif

#endif /* HML_KMEANS_GPU_H_INCLUDED_ */
