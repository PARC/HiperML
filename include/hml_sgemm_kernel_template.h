#ifndef HML_SGEMM_KERNEL_TEMPLATE_H_INCLUDED_
#define HML_SGEMM_KERNEL_TEMPLATE_H_INCLUDED_

extern texture<float, 1> texFloatA;
extern texture<float, 1> texFloatB;

static __inline__ __device__ float texA(const int &i) {
  return tex1Dfetch(texFloatA, i);
}

static __inline__ __device__ float texB(const int &i) {
  return tex1Dfetch(texFloatB, i);
}

/* Single-precision general matrix multiplication (sGEMM) kernel.
 * Both A & B are not transposed, and thus 'NN' function name suffix
 * we assume row-major order.
 */
template<int colStops, int rowStops, bool useTextureMem>
__global__ void
hmlSgemmVarKNN(float       *C,
               const float *A,
               const float *B,
               const int    M,
               const int    N,
               const int    K,
               const float  alpha,
               const float  beta) {
  /* thread row and column within Csub */
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 64 + tx;
  /* reshape the 64 x 4 grid into a 16 x 16 grid */
  int row = tid / 16;
  const int col = tid % 16;
  /* set base row and column of the Csub matrix within C */
  const int baseRow =
    (blockIdx.x + blockIdx.y * gridDim.x) * (rowStops * 16) + row;
  const int baseCol = blockIdx.z * (colStops * 16) + col;
  /* K16 is the number of A columns rounded down to the nearest
   * multiple of 16
   */
  const int K16 = K / 16 * 16;
  /* Each thread block computes one sub-matrix Csub of C
   * Each thread computes colStops * rowStops elements of Csub
   * with stride 16 by accumulating results into Cval
   */
  float Cval[colStops * rowStops];

  /* init Cval array */
#pragma unroll
  for(int i = 0; i < colStops * rowStops; ++i) {
    Cval[i] = 0.0;
  }

  __shared__ float As[rowStops * 16][17];
  __shared__ float Bs[16][colStops * 16];

  float AsCache[rowStops];  /* register file for As */
  float BsCache[colStops];  /* register file for Bs */

  /* loop over all the sub-matrices of A and B
   * multiply each pair of sub-matrices together
   * and accumulate the results
   */
  int k;
  for(k = 0; k < K16; k += 16) {
    /* Load Asub and Bsub from device memory to shared memory
     * Each thread loads rowStops elements of A, and
     * colStops elements of B
     */
    int baseIdxA = baseRow * K + k + col;
    int baseIdxB = (k + row) * N + baseCol; //__mul24() is slower!
#pragma unroll
    for(int i = 0; i < rowStops; ++i) {
      if(useTextureMem) {
        As[row + i * 16][col] = texA(baseIdxA + (__mul24(i, K) << 4));
      }
      else {
        As[row + i * 16][col] = A[baseIdxA + (__mul24(i, K) << 4)];
      }
    }

#pragma unroll
    for(int j = 0; j < colStops; ++j) {
      if(useTextureMem) {
        Bs[row][col + j * 16] = texB(baseIdxB + j * 16);
      }
      else {
        Bs[row][col + j * 16] = B[baseIdxB + j * 16];
      }
    }
    /* synchronize to make sure the sub-matrices are loaded
     * before starting the computation
     */
    __syncthreads();

    /* cache As and Bs into registers, multiply Asub and Bsub */
#pragma unroll
    for(int e = 0; e < 16; ++e) {
#pragma unroll
      for(int i = 0; i < rowStops; ++i) {
        AsCache[i] = As[row + i * 16][e];
      }
#pragma unroll
      for(int j = 0; j < colStops; ++j) {
        BsCache[j] = Bs[e][col + j * 16];
      }
#pragma unroll
      for(int i = 0; i < rowStops; ++i)
#pragma unroll
        for(int j = 0; j < colStops; ++j) {
          Cval[i * colStops + j] += AsCache[i] * BsCache[j];
        }
    }
    /* synchronize to make sure that the preceding
     * computation is done before loading two new
     * sub-matrices of A and B in the next iteration
     */
    __syncthreads();
  }
  /* take care of the irregular block, if any */
  if(k < K) {
    /* Load Asub and Bsub from device memory to shared memory
     * Each thread loads rowStops elements of A, and
     * colStops elements of B
     */
    int baseIdxA = baseRow * K + k + col;
    int baseIdxB = (k + row) * N + baseCol; //__mul24() is slower!
#pragma unroll
    for(int i = 0; i < rowStops; ++i) {
      if(useTextureMem) {
        As[row + i * 16][col] = texA(baseIdxA + (__mul24(i, K) << 4));
      }
      else {
        As[row + i * 16][col] = A[baseIdxA + (__mul24(i, K) << 4)];
      }
    }

#pragma unroll
    for(int j = 0; j < colStops; ++j) {
      if(useTextureMem) {
        Bs[row][col + j * 16] = texB(baseIdxB + j * 16);
      }
      else {
        Bs[row][col + j * 16] = B[baseIdxB + j * 16];
      }
    }
    /* synchronize to make sure the sub-matrices are loaded
     * before starting the computation
     */
    __syncthreads();

    /* cache As and Bs into registers, multiply Asub and Bsub */
    for(int e = 0; e < K - k; ++e) {
#pragma unroll
      for(int i = 0; i < rowStops; ++i) {
        AsCache[i] = As[row + i * 16][e];
      }
#pragma unroll
      for(int j = 0; j < colStops; ++j) {
        BsCache[j] = Bs[e][col + j * 16];
      }
#pragma unroll
      for(int i = 0; i < rowStops; ++i)
#pragma unroll
        for(int j = 0; j < colStops; ++j) {
          Cval[i * colStops + j] += AsCache[i] * BsCache[j];
        }
    }
    /* synchronize to make sure that the preceding
     * computation is done before loading two new
     * sub-matrices of A and B in the next iteration
     */
    __syncthreads();
  }
  /* write Csub to device memory in which each thread writes
   * rowStops x colStops element
   */
  const int baseIdxC = baseRow * N + baseCol;
#pragma unroll
  for(int i = 0; i < rowStops; ++i) {
#pragma unroll
    for(int j = 0; j < colStops; ++j) {
      if((baseRow + i * 16 < M) && (baseCol + j * 16 < N)) {
        int idx = baseIdxC + ((i * N + j) << 4);
        C[idx] = alpha * Cval[i * colStops + j] + beta * C[idx];
      }
    }
  }
}

/* macro version of hmlSgemmVarKNNSmemBytes(void) below */
#define HML_SGEMM_VAR_K_NN_SMEM_BYTES(colStops, rowStops) \
  (((rowStops) * 16 * 17 + 16 * (colStops) * 16) * cBytesPerFloat)

/* keep this template function in the hope that future versions of
 * the nvcc compiler can eliminate dead code, because template-based
 * implementation ensures 'if (..SmemBytes<...>() <= cHmlMaxSmemBytes)'
 * can be evaluated at compile time
 */
template<int colStops, int rowStops>
int
hmlSgemmVarKNNSmemBytes(void) {
  return (((rowStops) * 16 * 17 + 16 * (colStops) * 16) * sizeof(float));
}

template<int colStops, int rowStops, bool useTextureMem>
void
hmlSgemmKernelVarKNNSet(HmlSgemmKernelVarK varK[cHmlMaxStops+1][cHmlMaxStops+1]) {
  /* the if-statement below does NOT eliminate dead code, which
   * is the reason we still need to use macros:
   * #if SGEMM_..._SMEM_BYTES() <= cHmlMaxSmemBytes
   * to enclose this template function for dead code elimination purpose
   */
  if(hmlSgemmVarKNNSmemBytes<colStops, rowStops>() <= cHmlMaxSmemBytes) {
    varK[colStops][rowStops] =
      hmlSgemmVarKNN<colStops, rowStops, useTextureMem>;
  }
  else {
    varK[colStops][rowStops] = NULL;
  }
}

/* Single-precision general matrix multiplication (sGEMM) kernel.
 * Both A & B are not transposed, and thus 'NN' function name suffix
 * we assume row-major order.
 */
template<int K, int colStops, int rowStops, bool useTextureMem>
__global__ void
hmlSgemmConstKNN(float       *C,
                 const float *A,
                 const float *B,
                 const int    M,
                 const int    N,
                 const float  alpha,
                 const float  beta) {
  /* thread row and column within Csub */
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 64 + tx;
  /* reshape the 64 x 4 grid into a 16 x 16 grid */
  const int row = tid / 16;
  const int col = tid % 16;
  /* row0 is the row # in A[] for thread (0, 0) */
  const int row0 = (blockIdx.x + blockIdx.y * gridDim.x) * (rowStops * 16);
  /* col0 is the column # in B[] for thread (0, 0) */
  const int col0 = blockIdx.z * (colStops * 16);

  __shared__ float As[rowStops * 16][K + ((K % 2 == 0) ? 1 : 0)];
  __shared__ float Bs[K][colStops * 16];

  /* Each thread block computes one sub-matrix Csub of C
   * Each thread computes colStops * rowStops elements of Csub
   * with stride 16 by accumulating results into Cval
   */
  float Cval[colStops * rowStops];
  /* init Cval array */
#pragma unroll
  for(int i = 0; i < colStops * rowStops; ++i) {
    Cval[i] = 0.0;
  }

  int   idx;
  /* load Bsub into shared memory Bs[] */
  /* BsubCols: number of VALID columns of Bsub[] */
  const int BsubCols = (N - col0 < colStops * 16) ? N - col0 : colStops * 16;
  idx = tid;
  //while (idx < colStops * 16 * K) {
  while(idx < BsubCols * K) {
    if(useTextureMem)
      Bs[idx / BsubCols][idx % BsubCols] =
        texB(col0 + idx / BsubCols * N + idx % BsubCols);
    else
      Bs[idx / BsubCols][idx % BsubCols] =
        B[col0 + idx / BsubCols * N + idx % BsubCols];
    idx += 256;
  }
  /* load Asub into shared memory As[] */
  /* index of Asub[0] in A[] */
  const int Asub0Idx = row0 * K;
  idx = tid;
  while(idx < rowStops * 16 * K) {
    if(useTextureMem) {
      As[idx / K][idx % K] = texA(Asub0Idx + idx);
    }
    else {
      As[idx / K][idx % K] = A[Asub0Idx + idx];
    }
    idx += 256;
  }
  /* synchronize to make sure both B and Asub are loaded
   * before starting the computation
   */
  __syncthreads();

  /* cache As and Bs into registers, multiply Asub and Bsub */
#pragma unroll
  for(int e = 0; e < K; ++e) {
    float AsCache[rowStops];  /* register file for As */
    float BsCache[colStops];  /* register file for Bs */
#pragma unroll
    for(int i = 0; i < rowStops; ++i) {
      AsCache[i] = As[row + i * 16][e];
    }
#pragma unroll
    for(int j = 0; j < colStops; ++j) {
      BsCache[j] = Bs[e][col + j * 16];
    }
#pragma unroll
    for(int i = 0; i < rowStops; ++i)
#pragma unroll
      for(int j = 0; j < colStops; ++j) {
        Cval[i * colStops + j] += AsCache[i] * BsCache[j];
      }
  }
  const int baseRow = row0 + row;
  const int baseCol = col0 + col;
  const int baseIdxC = baseRow * N + baseCol;
  /* write Csub to device memory in which each thread writes
   * rowStops x colStops elements
   */
#pragma unroll
  for(int i = 0; i < rowStops; ++i) {
#pragma unroll
    for(int j = 0; j < colStops; ++j) {
      if((baseRow + i * 16 < M) && (baseCol + j * 16 < N)) {
        int idxC = baseIdxC + ((i * N + j) << 4);
        C[idxC] = alpha * Cval[i * colStops + j] + beta * C[idxC];
      }
    }
  }
}

/* macro version of hmlSgemmConstKNNSmemBytes(void) below */
#define HML_SGEMM_CONST_K_NN_SMEM_BYTES(K, colStops, rowStops)\
  (((rowStops) * 16 * ((K) + (((K) % 2 == 0) ? 1 : 0)) +       \
    (K) * (colStops) * 16) * cBytesPerFloat)

/* keep this template function in the hope that future versions of
 * the nvcc compiler can eliminate dead code, because template-based
 * implementation ensures 'if (..SmemBytes<...>() <= cHmlMaxSmemBytes)'
 * can be evaluated at compile time
 */
template<int K, int colStops, int rowStops>
int
hmlSgemmConstKNNSmemBytes(void) {
  return (((rowStops) * 16 * ((K) + (((K) % 2 == 0) ? 1 : 0)) +
           (K) * (colStops) * 16) * sizeof(float));
}

template<int K, int colStops, int rowStops, bool useTextureMem>
void
hmlSgemmKernelConstKNNSet(
  HmlSgemmKernelConstK constK[cHmlMaxSkinnyK+1][cHmlMaxStops+1][cHmlMaxStops+1]) {
  /* the if-statement below does NOT eliminate dead code, which
   * is the reason we still need to use macros:
   * #if SGEMM_..._SMEM_BYTES() <= cHmlMaxSmemBytes
   * to enclose this template function for dead code elimination purpose
   */
  if(hmlSgemmConstKNNSmemBytes<K, colStops, rowStops>() <= cHmlMaxSmemBytes) {
    constK[K][colStops][rowStops] =
      hmlSgemmConstKNN<K, colStops, rowStops, useTextureMem>;
  }
  else {
    constK[K][colStops][rowStops] = NULL;
  }
}

#endif /* HML_SGEMM_KERNEL_TEMPLATE_H_INCLUDED_ */
