/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_KMEANS_CPU_H_INCLUDED_
#define HML_KMEANS_CPU_H_INCLUDED_

#include "hml_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* generate a random number in [min, max)
 * that is, min inclusive and max exclusive
 */
uint32_t randomUInt32(uint32_t min, uint32_t max);

void hmlPrintDataMatrix(const float *pData,
                        uint32_t         numColumns,
                        uint32_t         numRows);

void hmlDebugPrintAsmnts(uint32_t *pAsmnts, uint32_t numPoints);

float hmlKmeansDistance(const float *p1, const float *p2, uint32_t numDims);

void hmlKmeansAssign(uint32_t        *pAsmnts,
                     const float *pRows,
                     uint32_t         numDims,
                     uint32_t         numRows,
                     const float *pCtrds,
                     uint32_t         numClusts);

void hmlKmeansUpdate(float       *pCtrds,
                     uint32_t        *pSizes,
                     uint32_t         numClusts,
                     const uint32_t  *pAsmnts,
                     const float *pRows,
                     uint32_t         numDims,
                     uint32_t         numRows);

float kMeansResidual(const float *pCtrds,
                     const float *pCtrdsPrev,
                     uint32_t         numDims,
                     uint32_t         numClusts);

/* pRows, pCtrds, pAsmnts, and pSizes must be
 * pre-allocated by the caller.
 */
void hmlKmeansCpu(float       *pCtrds,         /* numDims x numClusts */
                  uint32_t        *pSizes,         /* numClusts */
                  uint32_t        *pAsmnts,        /* numRows */
                  float       *pFinalResidual, /* return the final residual */
                  const float *pRows,          /* numDims x numRows */
                  uint32_t         numDims,
                  uint32_t         numRows,
                  uint32_t         numClusts,
                  uint32_t         numIters,
                  float        stopResidual);   /* termination residual */

void hmlKmeansPrintCluster(const float *pCtrds,
                           const uint32_t  *pSizes,
                           uint32_t         numDims,
                           uint32_t         numClusts,
                           const uint32_t  *pAsmnts,
                           uint32_t         numRows);

#ifdef __cplusplus
}
#endif

#endif /* HML_KMEANS_CPU_H_INCLUDED_ */
