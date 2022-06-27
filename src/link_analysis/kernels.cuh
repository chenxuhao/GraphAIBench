#pragma once
#include <cub/cub.cuh>
#include "graph_gpu.h"

typedef cub::BlockReduce<score_t, BLOCK_SIZE> BlockReduce;

__global__ void contrib(GraphGPU g, score_t *scores, score_t *outgoing_contrib) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < g.V()) outgoing_contrib[u] = scores[u] / g.get_degree(u);
}

__global__ void l1norm(int nv, score_t *scores, score_t *sums, 
                       float *diff, score_t base_score) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float local_diff = 0;
	if(u < nv) {
		score_t new_score = base_score + kDamp * sums[u];
		local_diff += fabs(new_score - scores[u]);
		scores[u] = new_score;
		sums[u] = 0;
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}

