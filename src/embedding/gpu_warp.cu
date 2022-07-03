// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "common_kernels.cuh"
#include "cuda_launch_config.hpp"

__global__ void compute_error(GraphGPU g, latent_t *latents, 
                              score_t lambda, score_t step, 
                              score_t *errors, score_t *squared_errors) {
  __shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

  const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for (int u = warp_id; u < g.V(); u += num_warps) {
    latent_t *u_latent = &latents[K*u];
    score_t *u_err = &errors[K*u];
    if (thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(u+thread_lane);
    auto row_begin = ptrs[warp_lane][0];
    auto row_end   = ptrs[warp_lane][1];
    for (auto offset = row_begin; offset < row_end; offset ++) {
      auto v = g.getEdgeDst(offset);
      auto v_latent = &latents[v*K];
      latent_t temp_p[(K-1)/WARP_SIZE + 1];
      latent_t temp_q[(K-1)/WARP_SIZE + 1];
      score_t estimate = 0;
      for (int i = 0; i < K; i += WARP_SIZE) {
        if (thread_lane+i < K) {
          int j = i/WARP_SIZE;
          temp_p[j] = u_latent[thread_lane+i];
          temp_q[j] = v_latent[thread_lane+i];
          estimate += temp_p[j] * temp_q[j];
        }
      }
      warp_reduce(estimate);
      /*
      estimate += SHFL_DOWN(estimate, 16);
      estimate += SHFL_DOWN(estimate, 8);
      estimate += SHFL_DOWN(estimate, 4);
      estimate += SHFL_DOWN(estimate, 2);
      estimate += SHFL_DOWN(estimate, 1);
      estimate = SHFL(estimate, 0);
      //*/
      score_t delta = g.getEdgeData(offset) - estimate;
      if (thread_lane == 0) squared_errors[u] += delta * delta;
      for (int i = 0; i < K; i += WARP_SIZE) {
        if (thread_lane+i < K) {
          int j = i/WARP_SIZE;
          u_err[thread_lane+i] += temp_q[j] * delta;
        }
      }
    }
  }
}

void SGDSolver(BipartiteGraph &g, std::vector<latent_t> &latents, int *h_ordering) {
  size_t memsize = print_device_info(0);
  auto nv = g.V();
  auto ne = g.E();
  //auto num_users = g.V(0);
  //auto num_items = g.V(1);
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (nv-1) / WARPS_PER_BLOCK + 1;
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(compute_error, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA CF (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  //for (size_t i = 0; i < latents.size(); i++) std::cout << latents[i] << "\n";

  latent_t *h_latents = &latents[0];
  latent_t *d_latents;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_latents, nv * K * sizeof(latent_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_latents, h_latents, nv * K * sizeof(latent_t), cudaMemcpyHostToDevice));
  score_t h_total_error = 0, *d_total_error, *d_errors, *squared_errors;
  CUDA_SAFE_CALL(cudaMalloc((void **)&squared_errors, nv * sizeof(score_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_error, sizeof(score_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_errors, nv * K * sizeof(score_t)));
  init_const_float<<<(nv*K-1)/nthreads+1, nthreads>>>(nv*K, 0.0, d_errors);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
  int iter = 0;
  do {
    ++iter;
    h_total_error = 0.0;
    init_const_float<<<(nv-1)/nthreads+1, nthreads>>>(nv, 0.0, squared_errors);
    CUDA_SAFE_CALL(cudaMemcpy(d_total_error, &h_total_error, sizeof(score_t), cudaMemcpyHostToDevice));
    compute_error<<<nblocks, nthreads>>>(gg, d_latents, lambda, step, d_errors, squared_errors);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    update_vertex<<<(nv-1)/nthreads+1, nthreads>>>(gg, lambda, step, d_latents, d_errors);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    rmse<<<(nv-1)/nthreads+1, nthreads>>>(nv, squared_errors, d_total_error);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(&h_total_error, d_total_error, sizeof(score_t), cudaMemcpyDeviceToHost));
    printf("iteration %d: RMSE error = %f\n", iter, sqrt(h_total_error/ne));
    //CUDA_SAFE_CALL(cudaMemcpy(h_latents, d_latents, nv*K*sizeof(latent_t), cudaMemcpyDeviceToHost));
  } while (iter < max_iters && h_total_error > cf_epsilon);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [gpu_base] = " << t.Seconds() << " sec\n";

  CUDA_SAFE_CALL(cudaMemcpy(h_latents, d_latents, nv*K*sizeof(latent_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_latents));
  CUDA_SAFE_CALL(cudaFree(d_total_error));
  CUDA_SAFE_CALL(cudaFree(d_errors));
  CUDA_SAFE_CALL(cudaFree(squared_errors));
}

