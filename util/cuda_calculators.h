// real_calculators.h
//
// last-edit-by: <> 
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#ifndef REAL_CALCULATORS_H
#define REAL_CALCULATORS_H 1

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  Parallel reduction kernels

  This file is https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu, as of 26.07.2022
*/

#define _CG_ABI_EXPERIMENTAL
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>

namespace cg = cooperative_groups;

static inline bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    mySum += __shfl_down_sync(mask, mySum, offset);
  }
  return mySum;
}

#if __CUDA_ARCH__ >= 800
// Specialize warpReduceFunc for int inputs to use __reduce_add_sync intrinsic
// when on SM 8.0 or higher
template <>
__device__ __forceinline__ int warpReduceSum<int>(unsigned int mask,
                                                  int mySum) {
  mySum = __reduce_add_sync(mask, mySum);
  return mySum;
}
#endif

/*
  Parallel sum reduction using shared memory
  - takes log(n) steps for n input elements
  - uses n threads
  - only works for power-of-2 arrays
*/

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void reduce0(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    // modulo arithmetic is slow!
    if ((tid % (2 * s)) == 0) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
template <class T>
__global__ void reduce1(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;

    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
  This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
  This version uses n/2 threads --
  it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}

/*
  This version uses the warp shuffle operation if available to reduce
  warp synchronization. When shuffle is not available the final warp's
  worth of work is unrolled to reduce looping overhead.

  See
  http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
  for additional information about using shuffle to perform a reduction
  within a warp.

  Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
  In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
  If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void reduce4(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockSize < n) mySum += g_idata[i + blockSize];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

/*
  This version is completely unrolled, unless warp shuffle is available, then
  shuffle is used within a loop.  It uses a template parameter to achieve
  optimal code for any (power of 2) number of threads.  This requires a switch
  statement in the host code to handle all the different thread block sizes at
  compile time. When shuffle is available, it is used to reduce warp
  synchronization.

  Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
  In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
  If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void reduce5(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockSize < n) mySum += g_idata[i + blockSize];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

/*
  This version adds multiple elements per thread sequentially.  This reduces
  the overall cost of the algorithm while keeping the work complexity O(n) and
  the step complexity O(log n). (Brent's Theorem optimization)

  Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
  In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
  If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const T *__restrict__ g_idata, T *__restrict__ g_odata,
                        unsigned int n) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;
  unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
  maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
  const unsigned int mask = (0xffffffff) >> maskLength;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum += g_idata[i];
      i += gridSize;
    }
  }

  // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
  // SM 8.0
  mySum = warpReduceSum<T>(mask, mySum);

  // each thread puts its local sum into shared memory
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = mySum;
  }

  __syncthreads();

  const unsigned int shmem_extent =
    (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
  const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    mySum = sdata[tid];
    // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum<T>(ballot_result, mySum);
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = mySum;
  }
}

// Performs a reduction step and updates numTotal with how many are remaining
template <typename T, typename Group>
__device__ T cg_reduce_n(T in, Group &threads) {
  return cg::reduce(threads, in, cg::plus<T>());
}

template <class T>
__global__ void cg_reduce(T *g_idata, T *g_odata, unsigned int n) {
  // Shared memory for intermediate steps
  T *sdata = SharedMemory<T>();
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Handle to tile in thread block
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

  unsigned int ctaSize = cta.size();
  unsigned int numCtas = gridDim.x;
  unsigned int threadRank = cta.thread_rank();
  unsigned int threadIndex = (blockIdx.x * ctaSize) + threadRank;

  T threadVal = 0;
  {
    unsigned int i = threadIndex;
    unsigned int indexStride = (numCtas * ctaSize);
    while (i < n) {
      threadVal += g_idata[i];
      i += indexStride;
    }
    sdata[threadRank] = threadVal;
  }

  // Wait for all tiles to finish and reduce within CTA
  {
    unsigned int ctaSteps = tile.meta_group_size();
    unsigned int ctaIndex = ctaSize >> 1;
    while (ctaIndex >= 32) {
      cta.sync();
      if (threadRank < ctaIndex) {
        threadVal += sdata[threadRank + ctaIndex];
        sdata[threadRank] = threadVal;
      }
      ctaSteps >>= 1;
      ctaIndex >>= 1;
    }
  }

  // Shuffle redux instead of smem redux
  {
    cta.sync();
    if (tile.meta_group_rank() == 0) {
      threadVal = cg_reduce_n(threadVal, tile);
    }
  }

  if (threadRank == 0) g_odata[blockIdx.x] = threadVal;
}

template <class T, size_t BlockSize, size_t MultiWarpGroupSize>
__global__ void multi_warp_cg_reduce(T *g_idata, T *g_odata, unsigned int n) {
  // Shared memory for intermediate steps
  T *sdata = SharedMemory<T>();
  __shared__ cg::experimental::block_tile_memory<sizeof(T), BlockSize> scratch;

  // Handle to thread block group
  auto cta = cg::experimental::this_thread_block(scratch);
  // Handle to multiWarpTile in thread block
  auto multiWarpTile = cg::experimental::tiled_partition<MultiWarpGroupSize>(cta);

  unsigned int gridSize = BlockSize * gridDim.x;
  T threadVal = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  int nIsPow2 = !(n & n-1);
  if (nIsPow2) {
    unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      threadVal += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + BlockSize) < n) {
        threadVal += g_idata[i + blockDim.x];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * BlockSize + threadIdx.x;
    while (i < n) {
      threadVal += g_idata[i];
      i += gridSize;
    }
  }

  threadVal = cg_reduce_n(threadVal, multiWarpTile);

  if (multiWarpTile.thread_rank() == 0) {
    sdata[multiWarpTile.meta_group_rank()] = threadVal;
  }
  cg::sync(cta);

  if (threadIdx.x == 0) {
    threadVal = 0;
    for (int i=0; i < multiWarpTile.meta_group_size(); i++) {
      threadVal += sdata[i];
    }
    g_odata[blockIdx.x] = threadVal;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void reduce(int size, int threads, int blocks, int whichKernel, T *d_idata,
            T *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
    (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  // as kernel 9 - multi_warp_cg_reduce cannot work for more than 64 threads
  // we choose to set kernel 7 for this purpose.
  if (threads < 64 && whichKernel == 9)
    {
      whichKernel = 7;
    }

  // choose which of the optimized versions of reduction to launch
  switch (whichKernel) {
  case 0:
    reduce0<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
    break;

  case 1:
    reduce1<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
    break;

  case 2:
    reduce2<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
    break;

  case 3:
    reduce3<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
    break;

  case 4:
    switch (threads) {
    case 512:
      reduce4<T, 512>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 256:
      reduce4<T, 256>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 128:
      reduce4<T, 128>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 64:
      reduce4<T, 64>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 32:
      reduce4<T, 32>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 16:
      reduce4<T, 16>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 8:
      reduce4<T, 8>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 4:
      reduce4<T, 4>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 2:
      reduce4<T, 2>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 1:
      reduce4<T, 1>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    }

    break;

  case 5:
    switch (threads) {
    case 512:
      reduce5<T, 512>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 256:
      reduce5<T, 256>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 128:
      reduce5<T, 128>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 64:
      reduce5<T, 64>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 32:
      reduce5<T, 32>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 16:
      reduce5<T, 16>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 8:
      reduce5<T, 8>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 4:
      reduce5<T, 4>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 2:
      reduce5<T, 2>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 1:
      reduce5<T, 1>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    }

    break;

  case 6:
    if (isPow2(size)) {
      switch (threads) {
      case 512:
	reduce6<T, 512, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 256:
	reduce6<T, 256, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 128:
	reduce6<T, 128, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 64:
	reduce6<T, 64, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 32:
	reduce6<T, 32, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 16:
	reduce6<T, 16, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 8:
	reduce6<T, 8, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 4:
	reduce6<T, 4, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 2:
	reduce6<T, 2, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 1:
	reduce6<T, 1, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;
      }
    } else {
      switch (threads) {
      case 512:
	reduce6<T, 512, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 256:
	reduce6<T, 256, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 128:
	reduce6<T, 128, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 64:
	reduce6<T, 64, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 32:
	reduce6<T, 32, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 16:
	reduce6<T, 16, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 8:
	reduce6<T, 8, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 4:
	reduce6<T, 4, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 2:
	reduce6<T, 2, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 1:
	reduce6<T, 1, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;
      }
    }

    break;

  case 7:
    // For reduce7 kernel we require only blockSize/warpSize
    // number of elements in shared memory
    smemSize = ((threads / 32) + 1) * sizeof(T);
    if (isPow2(size)) {
      switch (threads) {
      case 1024:
	reduce7<T, 1024, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;
      case 512:
	reduce7<T, 512, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 256:
	reduce7<T, 256, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 128:
	reduce7<T, 128, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 64:
	reduce7<T, 64, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 32:
	reduce7<T, 32, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 16:
	reduce7<T, 16, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 8:
	reduce7<T, 8, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 4:
	reduce7<T, 4, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 2:
	reduce7<T, 2, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 1:
	reduce7<T, 1, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;
      }
    } else {
      switch (threads) {
      case 1024:
	reduce7<T, 1024, true>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;
      case 512:
	reduce7<T, 512, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 256:
	reduce7<T, 256, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 128:
	reduce7<T, 128, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 64:
	reduce7<T, 64, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 32:
	reduce7<T, 32, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 16:
	reduce7<T, 16, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 8:
	reduce7<T, 8, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 4:
	reduce7<T, 4, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 2:
	reduce7<T, 2, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;

      case 1:
	reduce7<T, 1, false>
	  <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
	break;
      }
    }

    break;
  case 8:
    cg_reduce<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
    break;
  case 9:
    constexpr int numOfMultiWarpGroups = 2;
    smemSize = numOfMultiWarpGroups * sizeof(T);
    switch (threads) {
    case 1024:
      multi_warp_cg_reduce<T, 1024, 1024/numOfMultiWarpGroups>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 512:
      multi_warp_cg_reduce<T, 512, 512/numOfMultiWarpGroups>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 256:
      multi_warp_cg_reduce<T, 256, 256/numOfMultiWarpGroups>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 128:
      multi_warp_cg_reduce<T, 128, 128/numOfMultiWarpGroups>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 64:
      multi_warp_cg_reduce<T, 64, 64/numOfMultiWarpGroups>
	<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    default:
      printf("thread block size of < 64 is not supported for this kernel\n");
      break;
    }
    break;
  }
}

/*! \brief A function to convert input data from \p T to float and calculate its power in parallel on GPU
 * 
 * It is a template function to convert input data from \p T to float and calculate its power in parallel on GPU
 * \tparam T  The input data type
 *
 * The data type convertation is done with an overloadded function `scalar_typecast`
 * \see scalar_typecast
 *
 * The supported data convertation is shown in the following table (we can add more support here later)
 *
 * |T      |
 * |-------|
 * |double |
 * |half   |
 * |int    |
 * |int16_t|
 * |int8_t |
 *
 * \param[in]  d_data   Input data
 * \param[in]  ndata    Number of data
 * \param[out] d_float  Converted data in float
 * \param[out] d_float2 Power of converted data 
 *
 */
template <typename T>
__global__ void cudautil_pow(const T *d_data, float *d_float, float *d_float2, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(idx < ndata){
    float f_data;

    scalar_typecast(d_data[idx], f_data);
    d_float[idx]  = f_data;
    d_float2[idx] = f_data*f_data;
  }
}

template <typename T>
class RealMeanStddevCalculator {
  
 public:
  float mean;   ///< Mean of the difference between input two vectors, always in float
  float stddev; ///< Standard deviation of the difference between input two vectors, always in float

  
  //! Constructor of class RealMeanStddevCalculator
  /*!
   * - initialise the class
   * - create required device memory
   * - convert input data from \p T to float and calculate its power in a single CUDA kernel `cudautil_pow`
   * - reduce the float data and its power to get mean
   * - calculate standard deviation with the mean of float and power data
   *
   * \param[in] raw     The input vector on device/host with data type \p T
   * \param[in] ndata   Number of data
   * \param[in] nthread Number of threads per CUDA block to run kernel `cudautil_pow`
   * \param[in] method  Data reduction method, which can be from 0 to 7 inclusive
   *
   * As kernel `cudautil_pow` uses `scalar_typecast` to convert \p T to float, the support \p T can be
   *
   * |T |
   * |--|
   * |double | 
   * |half   |
   * |int    | 
   * |int16_t|
   * |int8_t |
   * 
   * \see cudautil_pow, reduce, scalar_typecast
   *
   */
 RealMeanStddevCalculator(T *raw, int ndata, int nthread, int method)
   :ndata(ndata), nthread(nthread), method(method){

    /* Sort out input buffers */
    data = copy2device(raw, ndata, type);
    
    // Now do calculation
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;
    
    checkCudaErrors(cudaMallocManaged(&d_float,  ndata*sizeof(float), cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&d_float2, ndata*sizeof(float), cudaMemAttachGlobal));
    
    checkCudaErrors(cudaMallocManaged(&d_reduction, nblock*sizeof(float), cudaMemAttachGlobal));
    
    cudautil_pow<<<nblock, nthread>>>(data, d_float, d_float2, ndata);
    getLastCudaError("Kernel execution failed [ cudautil_pow ]");
    
    // First reduce mean data
    reduce(ndata,  nthread, nblock, method, d_float, d_reduction);
    checkCudaErrors(cudaDeviceSynchronize());
    if(nblock > 1){
      reduce(nblock, nthread, 1, method, d_reduction, d_float);
      checkCudaErrors(cudaDeviceSynchronize());
      mean = d_float[0]/(float)ndata;
    }else{
      mean = d_reduction[0]/(float)ndata;
    }
    
    // Second reduce mean power 2 data
    reduce(ndata,  nthread, nblock, method, d_float2, d_reduction);
    checkCudaErrors(cudaDeviceSynchronize());
    if(nblock > 1){
      reduce(nblock, nthread, 1, method, d_reduction, d_float2);
      checkCudaErrors(cudaDeviceSynchronize());
      mean2 = d_float2[0]/(float)ndata;
    }else{
      mean2 = d_reduction[0]/(float)ndata;
    }

    // Got final numbers
    stddev = sqrtf(mean2 - mean*mean);

    // As we only need stddev and mean
    // Probably better to free all memory here
    checkCudaErrors(cudaFree(d_float));
    checkCudaErrors(cudaFree(d_float2));
    checkCudaErrors(cudaFree(d_reduction));
    
    remove_device_copy(type, data);
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealMeanStddevCalculator class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealMeanStddevCalculator(){
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
 private:
  enum cudaMemoryType type; ///< memory type

  int ndata; ///< Number of input data
  int nthread; ///< Number of threads per CUDA block
  int nblock;  ///< Number of CUDA blocks
  int method; ///< data d_reduction method
  
  T *data = NULL;
  float *d_float = NULL;
  float *d_float2 = NULL;

  float *d_reduction; ///< it holds intermediate float data duration data d_reduction on device
  
  float mean2; ///< mean of difference power of 2
};

template <typename TMIN, typename TSUB, typename TRES>
__device__ static inline void scalar_subtract(const TMIN minuend, const TSUB subtrahend, TRES &result) {
  TRES casted_minuend;
  TRES casted_subtrahend;
  
  scalar_typecast(minuend,    casted_minuend);
  scalar_typecast(subtrahend, casted_subtrahend);
  
  result = casted_minuend - casted_subtrahend;
}


/*! \brief Overloadded kernel to get d_difference between two real input vectors
 *
 * \tparam T1 Data type of the first input vector
 * \tparam T2 Data type of the second input vector
 * 
 * \param[in]  d_data1 The first input vector in \p T1
 * \param[in]  d_data2 The second input vector in \p T2
 * \param[in]  ndata   Number of data
 * \param[out] d_diff  The d_difference between these two vectors in float, it is always in float
 *
 * The kernel uses `scalar_subtract` to get difference (in float) between two numbers and currently it supports (we can add more support later).
 *
 * T1     | T2
 * -------|----
 * float  | float
 * float  | half
 * half   | float 
 * half   | half 
 * 
 * \see scalar_subtract
 * 
 */
template <typename T1, typename T2>
__global__ void cudautil_subtract(const T1 *d_data1, const T2 *d_data2, float *d_diff, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;

  if(idx < ndata){
    //d_diff[idx] = d_data1[idx] - d_data2[idx];

    scalar_subtract(d_data1[idx], d_data2[idx], d_diff[idx]);
  }
}

/*! \brief A class to get the difference between two real vectors
 *
 * \tparam T1 Typename of the data in one vector
 * \tparam T2 Typename of the data in the other vector
 *
 * 
 * Suggested combinations of T1 and T2 are (other combinations may not work, we can add more support later)
 * T1     | T2
 * -------|----
 * float  | float
 * float  | half
 * half   | float 
 * half   | half  
 * 
 * The class to get difference between two real vectors, it is allowed to have different types for these inputs and
 * the result will be in float.
 * 
 */
template <typename T1, typename T2>
class RealDifferentiator {

public:
  float *data  = NULL;  ///< the difference between input \p data1 and \p data2
  
  //! Constructor of RealDifferentiator class.
  /*!
   * 
   * - initialise the class
   * - create device memory for the difference \p diff
   * - calculate the difference with a CUDA kernel `cudautil_subtract`
   *
   * \see cudautil_subtract, scalar_subtract
   * 
   * \param[in] raw1 The first input real vector
   * \param[in] raw2 The second input real vector
   * \param[in] ndata   Number of data to subtract
   * \param[in] nthread Number of threads per CUDA block to run `cudautil_subtract`
   *
   */
  RealDifferentiator(T1 *raw1, T2 *raw2, int ndata, int nthread)
    :ndata(ndata), nthread(nthread){

    // sort out input buffers
    data1 = copy2device(raw1, ndata, type1);
    data2 = copy2device(raw2, ndata, type2);

    // Create output buffer as managed
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(float), cudaMemAttachGlobal));
    
    // setup kernel size and run it to get difference
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;
    cudautil_subtract<<<nblock, nthread>>>(data1, data2, data, ndata);
    getLastCudaError("Kernel execution failed [ cudautil_subtract ]");

    // Free intermediate memory
    remove_device_copy(type1, data1);
    remove_device_copy(type2, data2);
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealDifferentiator class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealDifferentiator(){
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
private:
  enum cudaMemoryType type1; ///< memory type
  enum cudaMemoryType type2; ///< memory type

  int ndata; ///< Number of input data
  int nthread; ///< Number of threads per CUDA block
  int nblock;  ///< Number of CUDA blocks

  T1 *data1 = NULL; ///< private variable to hold input vector pointer 1
  T2 *data2 = NULL; ///< private variable to hold input vector pointer 2
};


//! A template kernel to calculate phase and amplitude of input array 
/*!
 * 
 * \see scalar_typecast
 *
 * \tparam T Complex number component data type
 * 
 * \param[in]  v         input Complex data
 * \param[in]  ndata     Number of data samples to be calculated
 * \param[out] amplitude Calculated amplitude
 * \param[out] phase     Calculated amplitude
 *
 */
template <typename T>
__global__ void cudautil_amplitude_phase(const T *v, float *amplitude, float *phase, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(idx < ndata){
    // We always do calculation in float
    float v1;
    float v2;
    
    scalar_typecast(v[idx].x, v1);
    scalar_typecast(v[idx].y, v2);
    
    amplitude[idx] = sqrtf(v1*v1+v2*v2);
    phase[idx]     = atan2f(v2, v1); // in radians
  }
}

template <typename T>
class AmplitudePhaseCalculator{
 public:
  float *amp = NULL;///< Calculated amplitude on device
  float *pha = NULL;///< Calculated phase on device
  
  //! Constructor of AmplitudePhaseCalculator class.
  /*!
   * 
   * - initialise the class
   * - create device memory for amplitude and phase
   * - calculate phase and amplitude with CUDA
   *
   * \see cudautil_amplitude_phase
   *
   * \tparam TIN Input data type
   * 
   * \param[in] raw  input Complex data
   * \param[in] ndata   Number of samples to be converted, the size of data is 2*ndata
   * \param[in] nthread Number of threads per CUDA block to run `cudautil_amplitude_phase` kernel
   *
   */
 AmplitudePhaseCalculator(T *raw,
			  int ndata,
			  int nthread
			  )
   :ndata(ndata), nthread(nthread){

    // sourt out input data
    data = copy2device(raw, ndata, type);
    
    // Get output buffer as managed
    checkCudaErrors(cudaMallocManaged(&amp, ndata * sizeof(float), cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&pha, ndata * sizeof(float), cudaMemAttachGlobal));
  
    // Get amplitude and phase
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;
    cudautil_amplitude_phase<<<nblock, nthread>>>(data, amp, pha, ndata);

    remove_device_copy(type, data);
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealGeneratorNormal class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~AmplitudePhaseCalculator(){
    
    checkCudaErrors(cudaFree(amp));
    checkCudaErrors(cudaFree(pha));

    checkCudaErrors(cudaDeviceSynchronize());
  }

 private:
  int ndata; ///< number of values as a private parameter
  int nblock; ///< Number of CUDA blocks
  int nthread; ///< number of threas per block
  
  enum cudaMemoryType type; ///< memory type
  
  T *data; ///< To get hold on the input data
};

#endif // REAL_CALCULATORS_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
