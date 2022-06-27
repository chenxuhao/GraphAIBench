// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "kernels.cuh"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
//#define ENABLE_LB

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

__global__ void push_base(GraphGPU g, score_t *scores, score_t *sums) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < g.V()) {
		auto row_begin = g.edge_begin(src);
		auto row_end = g.edge_end(src);
		score_t value = scores[src] / score_t(g.get_degree(src));
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = g.getEdgeDst(offset);
			atomicAdd(&sums[dst], value);
		}
	}
}

__device__ __forceinline__ void expandByCta(GraphGPU g, const score_t *scores, score_t *sums, int *processed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int owner;
	__shared__ int sh_vertex;
	owner = -1;
	int size = 0;
	if(id < g.V()) {
		size = g.get_degree(id);
	}
	while(true) {
		if(size > BLOCK_SIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1) break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_vertex = id;
			processed[id] = 1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		int row_begin = g.edge_begin(sh_vertex);
		int row_end = g.edge_end(sh_vertex);
		int neighbor_size = row_end - row_begin;
		score_t value = scores[sh_vertex] / (score_t)neighbor_size;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int dst = g.getEdgeDst(edge);
				atomicAdd(&sums[dst], value);
			}
		}
	}
}

__device__ __forceinline__ void expandByWarp(GraphGPU g, const score_t *scores, score_t *sums, int *processed) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	if(id < g.V() && !processed[id]) {
		size = g.get_degree(id);
	}
	while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = id;
			processed[id] = 1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_vertex[warp_id];
		int row_begin = g.edge_begin(winner);
		int row_end = g.edge_end(winner);
		int neighbor_size = row_end - row_begin;
		score_t value = scores[winner] / (score_t)neighbor_size;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int edge = row_begin + i;
			if(i < neighbor_size) {
				int dst = g.getEdgeDst(edge);
				atomicAdd(&sums[dst], value);
			}
		}
	}
}

__global__ void push_lb(GraphGPU g, score_t *scores, score_t *sums, int *processed) {
	expandByCta(g, scores, sums, processed);
	expandByWarp(g, scores, sums, processed);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int src = tid;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[BLOCK_SIZE];
	__shared__ int src_idx[BLOCK_SIZE];
	__shared__ score_t values[BLOCK_SIZE];
	gather_offsets[tx] = 0;
	src_idx[tx] = 0;
	values[tx] = 0;
	__syncthreads();

	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if (tid < g.V() && !processed[tid]) {
		neighbor_offset = g.edge_begin(tid);
		neighbor_size = g.get_degree(tid);
		values[tx] = scores[src] / (score_t)neighbor_size;
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	
	int done = 0;
	int neighbors_done = 0;
	while (total_edges > 0) {
		__syncthreads();
		int i;
		for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < BLOCK_SIZE; i++) {
			int j = scratch_offset + i - done;
			gather_offsets[j] = neighbor_offset + neighbors_done + i;
			src_idx[j] = tx;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		if(tx < total_edges) {
			int edge = gather_offsets[tx];
			int dst = g.getEdgeDst(edge);
			atomicAdd(&sums[dst], values[src_idx[tx]]);
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

void PRSolver(Graph &g, score_t *h_scores) {
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (nv-1)/nthreads+1;
  assert(nblocks < 65536);
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(push_base, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA PageRank (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

	score_t *d_scores;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, nv * sizeof(score_t)));
	CUDA_SAFE_CALL(cudaMemcpy(d_scores, h_scores, nv * sizeof(score_t), cudaMemcpyHostToDevice));
	score_t *d_sums;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sums, nv * sizeof(score_t)));
	float *d_diff, h_diff;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));

#ifdef ENABLE_LB
	int *d_processed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_processed, nv * sizeof(int)));
#endif

	int iter = 0;
	const score_t base_score = (1.0f - kDamp) / nv;
	initialize <<<nblocks, nthreads>>> (nv, d_sums);
	printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	do {
		++ iter;
		h_diff = 0;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
#ifdef ENABLE_LB
		CUDA_SAFE_CALL(cudaMemset(d_processed, 0, nv * sizeof(int)));
		push_lb<<<nblocks, nthreads>>>(gg, d_scores, d_sums, d_processed);
#else
		push_base<<<nblocks, nthreads>>>(gg, d_scores, d_sums);
#endif
		l1norm <<<nblocks, nthreads>>> (nv, d_scores, d_sums, d_diff, base_score);
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
		printf(" %2d    %lf\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [gpu_push_base] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(ne) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, nv * sizeof(score_t), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sums));
	CUDA_SAFE_CALL(cudaFree(d_diff));
#ifdef ENABLE_LB
	CUDA_SAFE_CALL(cudaFree(d_processed));
#endif
	return;
}
