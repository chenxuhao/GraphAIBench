// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "kernels.cuh"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"

#define FUSED 0

typedef cub::BlockReduce<score_t, BLOCK_SIZE> BlockReduce;

__global__ void pull_step(GraphGPU g, score_t *sums, score_t *outgoing_contrib) {
	int dst = blockIdx.x * blockDim.x + threadIdx.x;
	if (dst < g.V()) {
    auto begin = g.edge_begin(dst);
    auto end = g.edge_end(dst);
		score_t incoming_total = 0;
		for (auto offset = begin; offset < end; ++ offset) {
			auto src = g.getEdgeDst(offset);
			//incoming_total += outgoing_contrib[src];
			incoming_total += __ldg(outgoing_contrib+src);
		}
		sums[dst] = incoming_total;
	}
}

// pull operation needs incoming neighbor list
__global__ void pull_fused(GraphGPU g, score_t *scores, score_t *outgoing_contrib, 
                           float *diff, score_t base_score) {
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float error = 0;
	if (src < g.V()) {
	  auto begin = g.edge_begin(src);
    auto end = g.edge_end(src);
		score_t incoming_total = 0;
		for (auto offset = begin; offset < end; ++ offset) {
			auto dst = g.getEdgeDst(offset);
			incoming_total += outgoing_contrib[dst];
		}
		score_t old_score = scores[src];
		scores[src] = base_score + kDamp * incoming_total;
		error += fabs(scores[src] - old_score);
	}
	float block_sum = BlockReduce(temp_storage).Sum(error);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void PRSolver(Graph &g, score_t *scores) {
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (nv-1)/nthreads+1;
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(pull_step, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA PageRank (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
 
	score_t *d_scores, *d_sums, *d_contrib;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, nv * sizeof(score_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, nv * sizeof(score_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, nv * sizeof(score_t)));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, nv * sizeof(score_t), cudaMemcpyHostToDevice));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));

	int iter = 0;
	const score_t base_score = (1.0f - kDamp) / nv;
	Timer t;
	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		contrib<<<nblocks, nthreads>>>(gg, d_scores, d_contrib);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
#if FUSED
		pull_fused<<<nblocks, nthreads>>>(gg, d_scores, d_contrib, d_diff, base_score);
#else
		pull_step<<<nblocks, nthreads>>>(gg, d_sums, d_contrib);
		l1norm<<<nblocks, nthreads>>>(nv, d_scores, d_sums, d_diff, base_score);
#endif
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [gpu_pull_base] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(ne) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
	CUDA_SAFE_CALL(cudaMemcpy(scores, d_scores, nv * sizeof(score_t), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}

