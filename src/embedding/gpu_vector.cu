// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#define USE_SHFL

#if __CUDACC_VER_MAJOR__ >= 9
#define SHFL_DOWN(a,b) __shfl_down_sync(0xFFFFFFFF,a,b)
#define SHFL(a,b) __shfl_sync(0xFFFFFFFF,a,b)
#else
#define SHFL_DOWN(a,b) __shfl_down(a,b)
#define SHFL(a,b) __shfl(a,b)
#endif
typedef cub::BlockReduce<score_t, BLOCK_SIZE> BlockReduce;

__global__ void update(int m, GraphGPU g, latent_t *latents, 
                       score_t lambda, score_t step, int *ordering, 
                       score_t *squared_errors) {
#ifndef USE_SHFL
  __shared__ score_t sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
#endif
  __shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

  const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for(int index = warp_id; index < m; index += num_warps) {
    //int user_id = ordering[index];
    int user_id = index;
    int base_p = user_id * K;
    auto user_lv = &latents[base_p];
    if(thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(user_id + thread_lane);
    const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[row];
    const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[row+1];
    //for(int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
    for(int offset = row_begin; offset < row_end; offset ++) {
      int item_id = g.getEdgeDst(offset);
      int base_q = item_id * K;
      auto item_lv = &latents[base_q];
      latent_t temp_p[K/WARP_SIZE + 1];
      latent_t temp_q[K/WARP_SIZE + 1];
      score_t estimate = 0;
      for (int i = 0; i < K; i += WARP_SIZE) {
        int j = i/WARP_SIZE;
        temp_p[j] = user_lv[thread_lane+i];
        temp_q[j] = item_lv[thread_lane+i];
        estimate += temp_p[j] * temp_q[j];
      }
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
      score_t delta = g.getEdgeData(offset) - estimate;
      if (thread_lane == 0) squared_errors[user_id] += delta * delta;
      for (int i = 0; i < K; i += WARP_SIZE) {
        int j = i/WARP_SIZE;
        latent_t new_user_feature = temp_p[j] + step * (-lambda * temp_p[j] + temp_q[j] * delta);
        latent_t new_item_feature = temp_q[j] + step * (-lambda * temp_q[j] + temp_p[j] * delta);
        user_lv[base_p+thread_lane+i] = new_user_feature;
        item_lv[base_q+thread_lane+i] = new_item_feature;
      }
    }
  }
  }

  __global__ void rmse(int m, score_t *squared_errors, score_t *total_error) {
    int uid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    score_t local_error = 0.0;
    if(uid < m) local_error = squared_errors[uid];
    score_t block_sum = BlockReduce(temp_storage).Sum(local_error);
    if(threadIdx.x == 0) atomicAdd(total_error, block_sum);
  }

  void SGDSolver(BipartiteGraph &g, latent_t *h_latents, int *h_ordering) {
    size_t memsize = print_device_info(0);
    auto nv = g.V();
    auto ne = g.E();
    auto num_users = g.V(0);
    auto num_items = g.V(1);
    auto md = g.get_max_degree();
    size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
    std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

    GraphGPU gg(g);
    size_t nthreads = BLOCK_SIZE;
    size_t nblocks = (num_users - 1) / WARPS_PER_BLOCK + 1;
    assert(nblocks < 65536);
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int max_blocks_per_SM = maximum_residency(update, nthreads, 0);
    std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
    //nblocks = std::min(max_blocks, nblocks);
    std::cout << "CUDA CF (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

    int *d_ordering;
    //CUDA_SAFE_CALL(cudaMalloc((void **)&d_ordering, num_users * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMemcpy(d_ordering, h_ordering, num_users * sizeof(int), cudaMemcpyHostToDevice));

    latent_t *d_latents;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_latents, nv * K * sizeof(latent_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_latents, h_latents, nv * K * sizeof(latent_t), cudaMemcpyHostToDevice));
    score_t h_error, *d_error, *squared_errors;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_error, sizeof(score_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&squared_errors, num_users * sizeof(score_t)));
    CUDA_SAFE_CALL(cudaMemset(d_error, 0, sizeof(score_t)));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    Timer t;
    t.Start();
    int iter = 0;
    do {
      ++iter;
      h_error = 0.0;
      CUDA_SAFE_CALL(cudaMemset(squared_errors, 0, num_users * sizeof(score_t)));
      CUDA_SAFE_CALL(cudaMemcpy(d_error, &h_error, sizeof(score_t), cudaMemcpyHostToDevice));
      update<<<nblocks, nthreads>>>(num_users, gg, d_latents, lambda, step, d_ordering, squared_errors);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      rmse<<<nblocks, nthreads>>>(num_users, squared_errors, d_error);
      CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(score_t), cudaMemcpyDeviceToHost));
      //printf("h_error=%f\n", h_error);
      printf("iteration %d: RMSE error = %f\n", iter, sqrt(h_error/ne));
      //CUDA_SAFE_CALL(cudaMemcpy(h_latents, nv * K * sizeof(latent_t), cudaMemcpyDeviceToHost));
      //print_latent_vector(nv, h_latents);
    } while (iter < max_iters && h_error > epsilon);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    t.Stop();

    CUDA_SAFE_CALL(cudaMemcpy(h_latents, d_latents, num_users * K * sizeof(latent_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_error));
    CUDA_SAFE_CALL(cudaFree(squared_errors));
  }

