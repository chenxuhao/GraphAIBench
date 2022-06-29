// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>

#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<score_t, BLOCK_SIZE> BlockReduce;

__global__ void update(vidType n, GraphGPU g, latent_t *latents, 
                       score_t lambda, score_t step, int *ordering, 
                       score_t *squared_errors) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < n) {
		//int user_id = ordering[tid];
		int user_id = tid;
		int row_begin = g.edge_begin(user_id);
		int row_end = g.edge_end(user_id); 
		int user_offset = K * user_id;
		latent_t *ulv = &latents[user_offset];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int item_id = g.getEdgeDst(offset);
			int item_offset = K * item_id;
			latent_t *ilv = &latents[item_offset];
			score_t estimate = 0;
			for (int i = 0; i < K; i++)
				estimate += ulv[i] * ilv[i];
			score_t delta = g.getEdgeData(offset) - estimate;
			squared_errors[user_id] += delta * delta;
			for (int i = 0; i < K; i++) {
				latent_t p_u = ulv[i];
				latent_t p_i = ilv[i];
				ulv[i] += step * (-lambda * p_u + p_i * delta);
				ilv[i] += step * (-lambda * p_i + p_u * delta);
			}
		}
	}
}

__global__ void rmse(int m, score_t *squared_errors, score_t *total_error) {
	int uid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	score_t local_error = 0.0;
	if (uid < m) local_error = squared_errors[uid];
	score_t block_sum = BlockReduce(temp_storage).Sum(local_error);
	if (threadIdx.x == 0) atomicAdd(total_error, block_sum);
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
	size_t nblocks = (num_users - 1) / nthreads + 1;
  if (nblocks > 65536) nblocks = 65536;
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
	  CUDA_SAFE_CALL(cudaDeviceSynchronize());
		CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(score_t), cudaMemcpyDeviceToHost));
		//printf("h_error=%f\n", h_error);
		printf("iteration %d: RMSE error = %f\n", iter, sqrt(h_error/ne));
		//CUDA_SAFE_CALL(cudaMemcpy(h_latents, d_latents, nv * K * sizeof(latent_t), cudaMemcpyDeviceToHost));
		//print_latent_vector(nv, h_latents);
	} while (iter < max_iters && h_error > cf_epsilon);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [gpu_base] = " << t.Seconds() << " sec\n";
 
	CUDA_SAFE_CALL(cudaMemcpy(h_latents, d_latents, nv * K * sizeof(latent_t), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_latents));
	CUDA_SAFE_CALL(cudaFree(d_error));
	CUDA_SAFE_CALL(cudaFree(squared_errors));
}

