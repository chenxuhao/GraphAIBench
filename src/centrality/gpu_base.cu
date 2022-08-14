// Copyright (c) 2020 MIT
// Xuhao Chen <cxh@mit.edu>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include "graph_gpu.h"
#include "worklist.cuh"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
typedef Worklist2<vidType, vidType> WLGPU;

__global__ void initialize(vidType m, int *depths) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < m) depths[id] = -1;
}

__global__ void insert(WLGPU in_queue, int src, int *path_counts, int *depths) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id == 0) {
    in_queue.push(src);
    path_counts[src] = 1;
    depths[src] = 0;
  }
  return;
}

__global__ void push_frontier(WLGPU in_queue, int *queue, int queue_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  vidType vertex;
  if (in_queue.pop_id(tid, vertex))
    queue[queue_len+tid] = vertex;
}

__global__ void bc_normalize(int m, score_t *scores, score_t max_score) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < m) scores[tid] = scores[tid] / (max_score);
}

// Shortest path calculation by forward BFS
__global__ void bc_forward(GraphGPU g, int depth, int *path_counts, int *depths, 
                           WLGPU in_queue, WLGPU out_queue) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  vidType src;
  if (in_queue.pop_id(tid, src)) {
    int row_begin = g.edge_begin(src);
    int row_end = g.edge_end(src); 
    for (int offset = row_begin; offset < row_end; ++ offset) {
      int dst = g.getEdgeDst(offset);
      if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1)) {
        assert(out_queue.push(dst));
      }
      if (depths[dst] == depth) {
        atomicAdd(&path_counts[dst], path_counts[src]);
      }
    }
  }
}

// Dependency accumulation by back propagation
__global__ void bc_reverse(int num, GraphGPU g, int depth,
                           const int *frontiers, const int *path_counts, 
                           const int *depths, score_t *deltas, score_t *scores) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int src = frontiers[tid];
    int row_begin = g.edge_begin(src);
    int row_end = g.edge_end(src); 
    score_t delta_src = 0;
    for (int offset = row_begin; offset < row_end; ++ offset) {
      int dst = g.getEdgeDst(offset);
      if (depths[dst] == depth + 1) {
        delta_src += static_cast<score_t>(path_counts[src]) / 
          static_cast<score_t>(path_counts[dst]) * (1 + deltas[dst]);
      }
    }
    deltas[src]  = delta_src;
    scores[src] += delta_src;
  }
}

void BCSolver(Graph &g, vidType source, score_t *h_scores) {
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (nv-1)/nthreads+1;
  //if (nblocks > 65536) nblocks = 65536;
  assert(nblocks < 65536);
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(bc_forward, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA Betweenness Centrality (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  score_t *d_scores, *d_deltas;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(score_t) * nv));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, sizeof(score_t) * nv));
  CUDA_SAFE_CALL(cudaMemset(d_scores, 0, nv * sizeof(score_t)));
  CUDA_SAFE_CALL(cudaMemset(d_deltas, 0, nv * sizeof(score_t)));
  int *d_path_counts, *d_depths, *d_frontiers;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_path_counts, sizeof(int) * nv));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, sizeof(int) * nv));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontiers, sizeof(int) * (nv+1)));
  CUDA_SAFE_CALL(cudaMemset(d_path_counts, 0, nv * sizeof(int)));

  int depth = 0;
  int nitems = 1;
  int frontiers_len = 0;
  vector<int> depth_index;
  depth_index.push_back(0);
  WLGPU wl1(nv), wl2(nv);
  WLGPU *inwl = &wl1, *outwl = &wl2;
  initialize <<<nblocks, nthreads>>> (nv, d_depths);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
  insert<<<1, 1>>>(*inwl, source, d_path_counts, d_depths);
  do {
    nblocks = (nitems - 1) / nthreads + 1;
    push_frontier<<<nblocks, nthreads>>>(*inwl, d_frontiers, frontiers_len);
    frontiers_len += nitems;
    depth_index.push_back(frontiers_len);
    printf("Forward: depth=%d, frontire_size=%d\n", depth, nitems);
    depth++;
    bc_forward<<<nblocks, nthreads>>>(gg, depth, d_path_counts, d_depths, *inwl, *outwl);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nitems = outwl->nitems();
    WLGPU *tmp = inwl;
    inwl = outwl;
    outwl = tmp;
    outwl->reset();
  } while (nitems > 0);
  for (int d = depth_index.size() - 2; d >= 0; d--) {
    nitems = depth_index[d+1] - depth_index[d];
    nblocks = (nitems - 1) / nthreads + 1;
    printf("Reverse: depth=%d, frontier_size=%d\n", d, nitems);
    bc_reverse<<<nblocks, nthreads>>>(nitems, gg, d, d_frontiers+depth_index[d], d_path_counts, d_depths, d_deltas, d_scores);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  score_t *d_max_score;
  d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + nv);
  score_t h_max_score;
  CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(score_t), cudaMemcpyDeviceToHost));
  nthreads = 512;
  nblocks = (nv - 1) / nthreads + 1;
  bc_normalize<<<nblocks, nthreads>>>(nv, d_scores, h_max_score);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "max_score = " << h_max_score << "\n";
  std::cout << "iterations = " << depth << "\n";
  std::cout << "runtime [bc_gpu_base] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(score_t) * nv, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_path_counts));
  CUDA_SAFE_CALL(cudaFree(d_depths));
  CUDA_SAFE_CALL(cudaFree(d_deltas));
  CUDA_SAFE_CALL(cudaFree(d_frontiers));
}

