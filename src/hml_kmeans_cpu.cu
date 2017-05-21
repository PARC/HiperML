#include <stdio.h>
#include <float.h>
/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#include "hml_kmeans_cpu.h"
#include <math.h>

/* generate a random number in [min, max)
 * that is, min inclusive and max exclusive
 */
uint32_t randomUInt32(uint32_t min, uint32_t max)
{
  return (min + rand() % (max - min));
}

void
hmlPrintDataMatrix(const float *pData, 
                uint32_t         numColumns, 
                uint32_t         numRows)
{
  uint32_t r;
  uint32_t c;
  
  fprintf(stderr, "Input matrix:\n");
  for (r = 0; r < numRows; ++r) {
    for (c = 0; c < numColumns; ++c)
      fprintf(stderr, " %f", *pData++);
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

void hmlDebugPrintAsmnts(uint32_t *pAsmnts, uint32_t numPoints)
{
  uint32_t row;
  
  fprintf(stderr, "\n");
  /* print out first 10 assignments */
  fprintf(stderr, "First 10 assignments:");
  for (row = 0; row < 10 && row < numPoints; ++row)
    fprintf(stderr, " %d", pAsmnts[row]);
  fprintf(stderr, "\n");
  /* print out last 10 assignments */
  if (numPoints > 10) {
    fprintf(stderr, "Last  10 assignments:");
    for (row = numPoints - 10; row < numPoints; ++row)
      fprintf(stderr, " %d", pAsmnts[row]);
  }
  fprintf(stderr, "\n");
}

float
hmlKmeansDistance(const float *p1, const float *p2, uint32_t numDims)
{
  float dist = 0.0;
  float delta;
  
  while (numDims--) {
    delta = *p1++ - *p2++;
    dist += delta * delta;
  }
  return dist;
}

void
hmlKmeansAssign(uint32_t        *pAsmnts,
             const float *pRows,
             uint32_t         numDims,
             uint32_t         numRows,
             const float *pCtrds,
             uint32_t         numClusts)
{
  uint32_t   k;           /* cluster id */
  uint32_t   row;
  uint32_t   minK;        /* min-hmlKmeansDistance cluster id */
  const float *pCurMean;
  const float *pCurRow;
  float  dist;
  float  minDist;

  /* avoid compiler warning */
  minK = numClusts;
  for (row = 0, pCurRow = pRows; row < numRows; ++row, pCurRow += numDims) {
    minDist = FLT_MAX;
    for (k = 0, pCurMean = pCtrds; k < numClusts; ++k, pCurMean += numDims) {
      dist = hmlKmeansDistance(pCurRow, pCurMean, numDims);
      if (dist < minDist) {
        minDist = dist;
        minK = k;
      }
    }
    pAsmnts[row] = minK;
  }
}

void
hmlKmeansUpdate(float       *pCtrds,             
             uint32_t        *pSizes,
             uint32_t         numClusts,
             const uint32_t  *pAsmnts,
             const float *pRows,
             uint32_t         numDims,
             uint32_t         numRows)
{
  uint32_t    k;   /* cluster id */
  uint32_t    dims;
  float  *pCurMean;
  
  /* reset memory pointed to by pCtrds */
  memset(pCtrds, 0, sizeof(float) * numDims * numClusts);
  memset(pSizes, 0, sizeof(uint32_t) * numClusts);

  /* sum along each dim for all points in the same cluster */
  while (numRows--) {
    k = (uint32_t)*pAsmnts++;
    ++pSizes[k];
    pCurMean = &pCtrds[numDims * k];
    dims = numDims;
    while (dims--) {
      *pCurMean += *pRows++;
      ++pCurMean;
    }
  }
  /* compute the centroids */
  for (k = 0, pCurMean = pCtrds; k < numClusts; ++k) {
    if (pSizes[k] == 0) {
      fprintf(stderr, "Error: Cluster #%d has no elements\n", k);
      exit(1);
    }
    dims = numDims;    
    while (dims--) {
      *pCurMean /= pSizes[k];
      ++pCurMean;
    }
  }
}

float
kMeansResidual(const float *pCtrds,
               const float *pCtrdsPrev,
               uint32_t         numDims,
               uint32_t         numClusts)
{
  float  residual = 0.0;
  float  delta;
  uint32_t   k;
  uint32_t   dim;
  
  for (k = 0; k < numClusts; ++k) {
    for (dim = 0; dim < numDims; ++dim) {
      //fprintf(stderr, "|%f - %f|  = %f\n", *pCtrds, 
      //  *pCtrdsPrev, fabs(*pCtrds - *pCtrdsPrev));
      delta = fabs(*pCtrds++ - *pCtrdsPrev++);
      residual = max(residual, delta);
    }
  }
  return residual;
}

/* pRows, pCtrds, pAsmnts, and pSizes must be
 * pre-allocated by the caller.
 */
void
hmlKmeansCpu(float       *pCtrds,         /* numDims x numClusts */
          uint32_t        *pSizes,         /* numClusts */
          uint32_t        *pAsmnts,        /* numRows */          
          float       *pFinalResidual, /* return the final residual */
          const float *pRows,          /* numDims x numRows */
          uint32_t         numDims,
          uint32_t         numRows,
          uint32_t         numClusts,
          uint32_t         numIters,
          float        stopResidual)   /* termination residual */          
{
  float  *pCtrdsPrev;
  float   residual = FLT_MAX;
  uint32_t    k;
  uint32_t    iter;
  
  MALLOC(pCtrdsPrev, float, numDims * numClusts);
  /* use Forgy method to initialize the cluster means */
  for (k = 0; k < numClusts; ++k) {
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
  /* perform numIters iterations, unless nothing is changed
   * during an assignment step
   */
  for (iter = 0; iter < numIters; ++iter) {
    hmlKmeansAssign(pAsmnts, pRows, numDims, numRows, pCtrds, numClusts);
#ifdef _DEBUG
    hmlDebugPrintAsmnts(pAsmnts, numRows);
#endif /* _DEBUG */

    /* save pCtrds to pCtrdsPrev */
    memcpy(pCtrdsPrev, pCtrds, sizeof(float) * numDims * numClusts);
    hmlKmeansUpdate(pCtrds, pSizes, numClusts, pAsmnts, pRows, numDims, numRows);
    residual = kMeansResidual(pCtrds, pCtrdsPrev, numDims, numClusts);
#ifdef _DEBUG
    fprintf(stderr, "Iteration #%d: residual = %f\n", iter + 1, residual);
#endif /* _DEBUG */
    if (residual <= stopResidual) {
      fprintf(stderr, "\nK-means CPU converged at iteration %d\n", iter + 1);
      break;
    }
  }
  *pFinalResidual = residual;
  FREE(pCtrdsPrev);
}

void
hmlKmeansPrintCluster(const float *pCtrds,
                   const uint32_t  *pSizes,
                   uint32_t         numDims,
                   uint32_t         numClusts,
                   const uint32_t  *pAsmnts,
                   uint32_t         numRows)
{
  uint32_t totalClusterSize = 0;
  uint32_t row;
  uint32_t k;
  uint32_t dim;
  
  fprintf(stdout, "\n");
  /* print out first 10 assignments */
  fprintf(stdout, "First 10 assignments:");
  for (row = 0; row < 10 && row < numRows; ++row)
    fprintf(stdout, " %d", pAsmnts[row]);
  fprintf(stdout, "\n");
  /* print out last 10 assignments */
  if (numRows > 10) {
    fprintf(stdout, "Last  10 assignments:");
    for (row = numRows - 10; row < numRows; ++row)
      fprintf(stdout, " %d", pAsmnts[row]);
  }
  fprintf(stdout, "\n\n");
  /* print out cluster sizes */
  for (k = 0; k < numClusts; ++k) {
    totalClusterSize += pSizes[k];
    fprintf(stdout, "Cluster #%d: size = %d\n     (", k, pSizes[k]);
    /* %.4f to allow comparison between cpu and gpu results */
    fprintf(stdout, "%.4f", *pCtrds++);
    for (dim = 1; dim < numDims; ++dim)
      fprintf(stdout, ", %.4f", *pCtrds++);
    fprintf(stdout, ")\n");
  }
  fprintf(stdout, "Total cluster size = %d\n", totalClusterSize);
  fprintf(stdout, "\n");
  if (totalClusterSize != numRows)
    fprintf(stderr, "Error: *** totalClusterSize != numRows ***\n");
}

