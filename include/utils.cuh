#pragma once
#include "cutil_subset.h"

__device__ __forceinline__ unsigned LaneId() {
  unsigned ret;
  asm("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

__global__ void init_const_float(int n, float value, float* array) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    array[i] = value;
  }
}

#define USE_SHFL

#if __CUDACC_VER_MAJOR__ >= 9
#define SHFL_DOWN(a,b) __shfl_down_sync(0xFFFFFFFF,a,b)
#define SHFL(a,b) __shfl_sync(0xFFFFFFFF,a,b)
#else
#define SHFL_DOWN(a,b) __shfl_down(a,b)
#define SHFL(a,b) __shfl(a,b)
#endif

inline __device__ void warp_reduce(float &estimate) {
#ifndef USE_SHFL
  __shared__ score_t sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
#endif

#ifdef USE_SHFL
  estimate += SHFL_DOWN(estimate, 16);
  estimate += SHFL_DOWN(estimate, 8);
  estimate += SHFL_DOWN(estimate, 4);
  estimate += SHFL_DOWN(estimate, 2);
  estimate += SHFL_DOWN(estimate, 1);
  estimate = SHFL(estimate, 0);
#else
  sdata[threadIdx.x] = estimate; __syncthreads();
  sdata[threadIdx.x] = estimate = estimate + sdata[threadIdx.x + 16]; __syncthreads();
  sdata[threadIdx.x] = estimate = estimate + sdata[threadIdx.x +  8]; __syncthreads();
  sdata[threadIdx.x] = estimate = estimate + sdata[threadIdx.x +  4]; __syncthreads();
  sdata[threadIdx.x] = estimate = estimate + sdata[threadIdx.x +  2]; __syncthreads();
  sdata[threadIdx.x] = estimate = estimate + sdata[threadIdx.x +  1]; __syncthreads();
  estimate = sdata[warp_lane*WARP_SIZE];
#endif
}
