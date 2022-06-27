// Copyright 2020
// Authors: Xuhao Chen <cxh@mit.edu>
#include "kernels.cuh"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"

//#define SHFL
//#define FUSED

__global__ void pull_step(GraphGPU g, score_t *sums, const score_t *outgoing_contrib) {
#ifndef SHFL
	__shared__ score_t sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
#endif
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int dst = warp_id; dst < g.V(); dst += num_warps) {
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = g.edge_begin(dst + thread_lane);
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[dst];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[dst+1];

		// compute local sum
		score_t sum = 0;
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			auto src = g.getEdgeDst(offset);
			sum += __ldg(outgoing_contrib+src);
		}
#ifndef SHFL
		// store local sum in shared memory,
		// and reduce local sums to global sum
		sdata[threadIdx.x] = sum; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
		if(thread_lane == 0) sums[dst] += sdata[threadIdx.x];
#else
		sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
		sum += __shfl_down_sync(0xFFFFFFFF, sum,  8);
		sum += __shfl_down_sync(0xFFFFFFFF, sum,  4);
		sum += __shfl_down_sync(0xFFFFFFFF, sum,  2);
		sum += __shfl_down_sync(0xFFFFFFFF, sum,  1);
		sum = __shfl_sync(0xFFFFFFFF, sum,  0);
		if(thread_lane == 0) sums[dst] = sum;
#endif
	}
}

// pull operation needs incoming neighbor list
__global__ void pull_fused(GraphGPU g, score_t *scores, score_t *outgoing_contrib, 
                           float *diff, score_t base_score) {
	__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ score_t sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	float error = 0;
	for(int dst = warp_id; dst < g.V(); dst += num_warps) {
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = g.edge_begin(dst + thread_lane);
		const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[dst];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[dst+1];

		// compute local sum
		score_t sum = 0;
		for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
			int src = g.getEdgeDst(offset);
			sum += outgoing_contrib[src];
		}
		// store local sum in shared memory
		sdata[threadIdx.x] = sum; __syncthreads();

		// reduce local sums to row sum
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		if(thread_lane == 0) {
			auto old_score = scores[dst];
			score_t new_score = base_score + kDamp * sdata[threadIdx.x];
			scores[dst] = new_score;
			error += fabs(new_score - old_score);
		}
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
  size_t nblocks = (nv-1)/WARPS_PER_BLOCK+1;
 
  score_t *d_scores, *d_sums, *d_contrib;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, nv * sizeof(score_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, nv * sizeof(score_t)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_contrib, nv * sizeof(score_t)));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, nv * sizeof(score_t), cudaMemcpyHostToDevice));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));

	int iter = 0;
	const score_t base_score = (1.0f - kDamp) / nv;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	const int nSM = deviceProp.multiProcessorCount;
#ifdef FUSED
	const int max_blocks_per_SM = maximum_residency(pull_fused, nthreads, 0);
#else
	const int max_blocks_per_SM = maximum_residency(pull_step, nthreads, 0);
#endif
	const int max_blocks = max_blocks_per_SM * nSM;
	const int mblocks = std::min(max_blocks, DIVIDE_INTO(nv, WARPS_PER_BLOCK));

	Timer t;
	t.Start();
	do {
		++iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
		contrib <<<nblocks, nthreads>>>(gg, d_scores, d_contrib);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
#ifdef FUSED
		pull_fused <<<mblocks, nthreads>>>(gg, d_scores, d_contrib, d_diff, base_score);
#else
		pull_step <<<mblocks, nthreads>>>(gg, d_sums, d_contrib);
		l1norm <<<nblocks, nthreads>>> (nv, d_scores, d_sums, d_diff, base_score);
#endif
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [gpu_pull_warp] = " << t.Seconds() << " sec\n";
	CUDA_SAFE_CALL(cudaMemcpy(scores, d_scores, nv * sizeof(score_t), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_contrib));
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}
