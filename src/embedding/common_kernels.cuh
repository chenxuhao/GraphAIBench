#include "graph_gpu.h"
#include "utils.cuh"
#include <cub/cub.cuh>
typedef cub::BlockReduce<score_t, BLOCK_SIZE> BlockReduce;

__global__ void rmse(int m, score_t *squared_errors, score_t *total_error) {
  int uid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  score_t local_error = 0.0;
  if(uid < m) local_error = squared_errors[uid];
  score_t block_sum = BlockReduce(temp_storage).Sum(local_error);
  if(threadIdx.x == 0) atomicAdd(total_error, block_sum);
}

__global__ void update_vertex(GraphGPU g, score_t lambda, score_t step, 
                              latent_t *latents, score_t *errors) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u < g.V()) {
    for (int i = 0; i < K; i++) {
      latents[K*u+i] += step * (-lambda * latents[K*u+i] + errors[K*u+i]);
      errors[K*u+i] = 0.0;
    }
  }
}

