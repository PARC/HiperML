#ifndef HML_SGEMV_KERNEL_TEMPLATE_H_INCLUDED_
#define HML_SGEMV_KERNEL_TEMPLATE_H_INCLUDED_

extern texture<float, 1> texFloatA;
extern texture<float, 1> texFloatx;

static __inline__ __device__ float texA(const int &i) {
  return tex1Dfetch(texFloatA, i);
}

static __inline__ __device__ float texx(const int &i) {
  return tex1Dfetch(texFloatx, i);
}

/* Single-precision general matrix multiplication (sGEMV) kernel.
 * Matrix A is not transposed, and thus 'N' function name suffix
 * we assume row-major order.
 */
template<int blockStops, bool useTextureMem>
__global__ void
hmlSgemvKernelBasicN(float       *y,
            const float *A,
            const float *x,
            const int    M,
            const int    N,
            const float  alpha,
            const float  beta)
{
  /* assuming one-dimensional thread block */
  const int tid = threadIdx.x;
  /* row is set to be the row # in A[] for this thread block */
  const int row = blockIdx.x + blockIdx.y * gridDim.x;
  const int baseIdxA = row * N;
  /* N16 is the number of A columns rounded down to the nearest
   * multiple of (blockStops * 16)
   */
  const int N16 = N / (blockStops * 16) * (blockStops * 16);

  /* each thread computes 1/(blockStops*16)-th sub-vector of y */
  __shared__ float yVal[blockStops * 16];
  /* needed for reduction without __syncthreads() */
  volatile float *yValInWarp;
  float ax = 0.0;

  if (row < M) {
    /* loop over all the sub-matrices of A and
     * the sub-vectors of x in stride 16
     * multiply each <sub-matrix, sub-vector> pair together
     * and accumulate the results
     */
    int n;
    for (n = 0; n < N16; n += (blockStops * 16)) {
      if (useTextureMem)
        ax += texA(baseIdxA + tid + n) * texx(tid + n);
      else
        ax += A[baseIdxA + tid + n] * x[tid + n];
    }
    if (tid + n < N) {
      if (useTextureMem)
        ax += texA(baseIdxA + tid + n) * texx(tid + n);
      else
        ax += A[baseIdxA + tid + n] * x[tid + n];
    }
    yVal[tid] = ax;  
    /* synchronize before reduction */
    __syncthreads();
    if (blockStops * 16 >= 512) {
      if (tid < 256)
        yVal[tid] += yVal[tid + 256];
      __syncthreads();
    }
    if (blockStops * 16 >= 256) {
      if (tid < 128)
        yVal[tid] += yVal[tid + 128];
      __syncthreads();
    }
    if (blockStops * 16 >= 128) {
      if (tid < 64)
        yVal[tid] += yVal[tid + 64];
      __syncthreads();
    }
    if (tid < 32) {
      yValInWarp = yVal;
      if (blockStops * 16 >= 64)
        yValInWarp[tid] += yValInWarp[tid + 32];
      if (blockStops * 16 >= 32)
        yValInWarp[tid] += yValInWarp[tid + 16];
      yValInWarp[tid] += yValInWarp[tid + 8];
      yValInWarp[tid] += yValInWarp[tid + 4];
      yValInWarp[tid] += yValInWarp[tid + 2];
      yValInWarp[tid] += yValInWarp[tid + 1];
    }
    /* write y[row] to device memory */
    if (tid == 0)
      y[row] = alpha * yVal[0] + beta * y[row];
  }
}

/* macro version of hmlSgemvKernelBasicNSmemBytes(void) below */
#define HML_SGEMV_BASIC_N_SMEM_BYTES(blockStops) \
  ((blockStops) * 16 * cBytesPerFloat)

/* keep this template function in the hope that future versions of
 * the nvcc compiler can eliminate dead code, because template-based
 * implementation ensures 'if (..SmemBytes<...>() <= cHmlMaxSmemBytes)'
 * can be evaluated at compile time
 */
template<int blockStops>
int
hmlSgemvKernelBasicNSmemBytes(void)
{
  return (blockStops * 16 * sizeof(float));
}

template<int blockStops, bool useTextureMem>
void
hmlSgemvKernelBasicNSet(HmlSgemvKernelVarN basic[cHmlMaxBlockStops+1])
{
  /* the if-statement below does NOT eliminate dead code, which
   * is the reason we still need to use macros:
   * #if HML_SGEMV_..._SMEM_BYTES() <= cHmlMaxSmemBytes
   * to enclose this template function for dead code elimination purpose
   */
  if (hmlSgemvKernelBasicNSmemBytes<blockStops>() <= cHmlMaxSmemBytes)
    basic[blockStops] = hmlSgemvKernelBasicN<blockStops, useTextureMem>;
  else
    basic[blockStops] = NULL;
}

/* Single-precision general matrix multiplication (sGEMV) kernel.
 * Matrix A is not transposed, and thus 'N' function name suffix
 * we assume row-major order.
 */
template<int blockStops, int rowStops, bool useTextureMem>
__global__ void
hmlSgemvKernelVarN(float       *y,
           const float *A,
           const float *x,
           const int    M,
           const int    N,
           const float  alpha,
           const float  beta)
{
  /* assuming one-dimensional thread block */
  const int tid = threadIdx.x;
  /* row # used for loading Asub */
  const int subRow = tid / 16;
  /* column # used for loading Asub */
  const int subCol = tid % 16;
  /* row is set to be the row # in A[] for thread 0 */
  int row = 
    (blockIdx.x + blockIdx.y * gridDim.x) * (rowStops * blockStops * 16);
  /* N16 is the number of A columns rounded down to the nearest
   * multiple of 16
   */
  const int N16 = N / 16 * 16;
  /* Each thread block computes one sub-vector of y
   * Each thread computes 'rowStops' elements of y
   * with stride 'blockStops' * 16 by accumulating results into yVal
   */
  float yVal[rowStops];

  /* init yVal array */
#pragma unroll
  for (int i = 0; i < rowStops; ++i)
    yVal[i] = 0.0;

  __shared__ float As[rowStops * blockStops * 16][17];
  __shared__ float xs[16];

  float AsCache[rowStops];  /* register file for As */
  float xsCache;            /* register for xs */

  /* loop over all the sub-matrices of A and
   * the sub-vectors of x in stride 16
   * multiply each <sub-matrix, sub-vector> pair together
   * and accumulate the results
   */
  int n;
  for (n = 0; n < N16; n += 16) {
    /* Load Asub and xsub from device memory to shared memory
     * Each thread loads rowStops elements of A and x
     */
    int baseIdxA = (row + subRow) * N + n + subCol;
#pragma unroll
    for (int i = 0; i < rowStops * 16; ++i) {
      if (useTextureMem)
        As[subRow + i * blockStops][subCol] =
          texA(baseIdxA + __mul24(i, N) * blockStops);
      else
        As[subRow + i * blockStops][subCol] =
          A[baseIdxA + __mul24(i, N) * blockStops];
    }
    /* only threads in the first row need to load xs from x */
    if (subRow == 0) {
      if (useTextureMem)
        xs[subCol] = texx(n + subCol);
      else
        xs[subCol] = x[n + subCol];
    }
    /* synchronize to make sure the sub-matrices are loaded
     * before starting the computation
     */
    __syncthreads();
  
    /* cache As and Bs into registers, multiply Asub and Bsub */
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      xsCache = xs[e];
#pragma unroll
      for (int i = 0; i < rowStops; ++i)
        AsCache[i] = As[tid + i * blockStops * 16][e];
#pragma unroll
      for (int i = 0; i < rowStops; ++i)
        yVal[i] += AsCache[i] * xsCache;
    }
    /* synchronize to make sure that the preceding
     * computation is done before loading two new
     * sub-matrices of A and B in the next iteration
     */
    __syncthreads();
  }
  if (n < N) {
    int baseIdxA = (row + subRow) * N + n + subCol;
#pragma unroll
    for (int i = 0; i < rowStops * 16; ++i) {
      if (useTextureMem)
        As[subRow + i * blockStops][subCol] =
          texA(baseIdxA + __mul24(i, N) * blockStops);
      else
        As[subRow + i * blockStops][subCol] =
          A[baseIdxA + __mul24(i, N) * blockStops];
    }
    /* only threads in the first row need to load xs from x */
    if (subRow == 0) {
      if (useTextureMem)
        xs[subCol] = texx(n + subCol);
      else
        xs[subCol] = x[n + subCol];
    }
    /* synchronize to make sure the sub-matrices are loaded
     * before starting the computation
     */
    __syncthreads();
  
    /* cache As and Bs into registers, multiply Asub and Bsub */
    for (int e = 0; e < N - n; ++e) {
      xsCache = xs[e];
#pragma unroll
      for (int i = 0; i < rowStops; ++i)
        AsCache[i] = As[tid + i * blockStops * 16][e];
#pragma unroll
      for (int i = 0; i < rowStops; ++i)
        yVal[i] += AsCache[i] * xsCache;
    }
    /* synchronize to make sure that the preceding
     * computation is done before loading two new
     * sub-matrices of A and B in the next iteration
     */
    __syncthreads();
  }
  row += tid;
  /* write y sub to device memory in which each thread writes
   * rowStops elements
   */
#pragma unroll
  for (int i = 0; i < rowStops; ++i) {
    int yIdx = row + i * blockStops * 16;
    if (yIdx < M)
      y[yIdx] = alpha * yVal[i] + beta * y[yIdx];
  }
}

/* macro version of hmlSgemvKernelVarNSmemBytes(void) below */
#define HML_SGEMV_VAR_N_SMEM_BYTES(blockStops, rowStops) \
  (((rowStops)*(blockStops)*16*17 + 16) * cBytesPerFloat)

/* keep this template function in the hope that future versions of
 * the nvcc compiler can eliminate dead code, because template-based
 * implementation ensures 'if (..SmemBytes<...>() <= cHmlMaxSmemBytes)'
 * can be evaluated at compile time
 */
template<int blockStops, int rowStops>
int
hmlSgemvKernelVarNSmemBytes(void)
{
  return ((rowStops * blockStops * 16 * 17 + 16) * sizeof(float));
}

template<int blockStops, int rowStops, bool useTextureMem>
void
hmlSgemvKernelSetVarNN(HmlSgemvKernelVarN varN[cHmlMaxBlockStops+1][cHmlMaxStops+1])
{
  /* the if-statement below does NOT eliminate dead code, which
   * is the reason we still need to use macros:
   * #if HML_SGEMV_..._SMEM_BYTES() <= cHmlMaxSmemBytes
   * to enclose this template function for dead code elimination purpose
   */
  if (hmlSgemvKernelVarNSmemBytes<blockStops, rowStops>() <= cHmlMaxSmemBytes) {
    varN[blockStops][rowStops] =
      hmlSgemvKernelVarN<blockStops, rowStops, useTextureMem>;
  }
  else
    varN[blockStops][rowStops] = NULL;
}

/* Single-precision general matrix multiplication (sGEMV) kernel.
 * Matrix A is not transposed, and thus 'N' function name suffix
 * we assume row-major order.
 */
template<int N, int blockStops, int rowStops, bool useTextureMem>
__global__ void
hmlSgemvKernelConstNN(float       *y,
             const float *A,
             const float *x,
             const int    M,
             const float  alpha,
             const float  beta)
{
  /* assuming one-dimensional thread block */
  const int tid = threadIdx.x;
  /* row is initially set to be the row # in A[] for thread #0 */
  int row =
    (blockIdx.x + blockIdx.y * gridDim.x) * (rowStops * blockStops * 16);
  
  __shared__ float As[rowStops*blockStops*16][N + ((N % 2 == 0) ? 1 : 0)];
  __shared__ float xs[N];
  
  /* Each thread block computes one sub-vector of y
   * Each thread computes 'rowStops' elements of y
   * with stride 'blockStops' * 16 by accumulating results into yVal
   */
  float yVal[rowStops];
  /* init yVal array */
#pragma unroll
  for (int i = 0; i < rowStops; ++i)
    yVal[i] = 0.0;

  int idx;  
  /* load vector x into shared memory xs[] */
  idx = tid;
  while (idx < N) {
    if (useTextureMem)
      xs[idx] = texx(idx);
    else
      xs[idx] = x[idx];
    idx += blockStops * 16;
  }
  /* load Asub into shared memory As[] */
  /* index of Asub[0] in A[] */
  const int Asub0Idx = row * N;
  idx = tid;  
  while (idx < rowStops * blockStops * 16 * N) {
    if (useTextureMem)
      As[idx / N][idx % N] = texA(Asub0Idx + idx);
    else
      As[idx / N][idx % N] = A[Asub0Idx + idx];
    idx += blockStops * 16;
  }
  /* synchronize to make sure both B and Asub are loaded 
   * before starting the computation
   */
  __syncthreads();  
  
  /* cache As and xs into registers, multiply As and xs */
#pragma unroll
  for (int e = 0; e < N; ++e) {
    float AsCache[rowStops];  /* register file for As */
    float xsCache = xs[e];    /* register for xs */
#pragma unroll
    for (int i = 0; i < rowStops; ++i)
      AsCache[i] = As[tid + i * blockStops * 16][e];
#pragma unroll
    for (int i = 0; i < rowStops; ++i)
      yVal[i] += AsCache[i] * xsCache;
  }  
  row += tid;
  /* write y sub to device memory in which each thread writes
   * rowStops elements
   */
#pragma unroll
  for (int i = 0; i < rowStops; ++i) {
    int yIdx = row + i * blockStops * 16;
    if (yIdx < M)
      y[yIdx] = alpha * yVal[i] + beta * y[yIdx];
  }
}

/* macro version of hmlSgemvKernelConstNNSmemBytes(void) below */
#define HML_SGEMV_CONST_N_N_SMEM_BYTES(N, blockStops, rowStops) \
  (((rowStops) * (blockStops) * 16 * ((N) + (((N) % 2 == 0) ? 1 : 0)) + \
    (N)) * cBytesPerFloat)

/* keep this template function in the hope that future versions of
 * the nvcc compiler can eliminate dead code, because template-based
 * implementation ensures 'if (..SmemBytes<...>() <= cHmlMaxSmemBytes)'
 * can be evaluated at compile time
 */
template<int N, int blockStops, int rowStops>
int
hmlSgemvKernelConstNNSmemBytes(void)
{
  return ((rowStops * blockStops * 16 * (N + ((N % 2 == 0) ? 1 : 0)) +
           N) * sizeof(float));
}

template<int N, int blockStops, int rowStops, bool useTextureMem>
void
hmlSgemvKernelSetConstNN(
  HmlSgemvKernelConstN constN[cHmlMaxSkinnyN+1][cHmlMaxBlockStops+1][cHmlMaxStops+1])
{
  /* the if-statement below does NOT eliminate dead code, which
   * is the reason we still need to use macros:
   * #if HML_SGEMV_..._SMEM_BYTES() <= cHmlMaxSmemBytes
   * to enclose this template function for dead code elimination purpose
   */
  if (hmlSgemvKernelConstNNSmemBytes<N, blockStops, rowStops>() <=
      cHmlMaxSmemBytes) {
    constN[N][blockStops][rowStops] =                                     
      hmlSgemvKernelConstNN<N, blockStops, rowStops, useTextureMem>;
  }
  else
    constN[N][blockStops][rowStops] = NULL;    
}

#endif /* HML_SGEMV_KERNEL_TEMPLATE_H_INCLUDED_ */
