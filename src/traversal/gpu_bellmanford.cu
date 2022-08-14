// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph_gpu.h"
#include "worklist.cuh"
#include "cuda_launch_config.hpp"
typedef Worklist2<vidType, vidType> WLGPU;

__global__ void bellman_ford(GraphGPU g, elabel_t *dist, WLGPU in_frontier, WLGPU out_frontier) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  vidType src;
  if(in_frontier.pop_id(tid, src)) {
    auto row_begin = g.edge_begin(src);
    auto row_end = g.edge_end(src);
    for (auto offset = row_begin; offset < row_end; ++ offset) {
      auto dst = g.getEdgeDst(offset);
      elabel_t old_dist = dist[dst];
      elabel_t new_dist = dist[src] + g.getEdgeData(offset);
      if (new_dist < old_dist) {
        if (atomicMin(&dist[dst], new_dist) > new_dist)
          out_frontier.push(dst);
      }
    }
  }
}

__global__ void insert(vidType source, WLGPU in_frontier) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id == 0) in_frontier.push(source);
  return;
}

void SSSPSolver(Graph &g, vidType source, elabel_t *h_dist, int delta) {
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
  auto max_blocks_per_SM = maximum_residency(bellman_ford, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA SSSP Bellman-Ford baseline (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  elabel_t zero = 0;
  elabel_t * d_dist;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, nv * sizeof(elabel_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, nv * sizeof(elabel_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(&d_dist[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  WLGPU wl1(nv), wl2(nv);
  WLGPU *in_frontier = &wl1, *out_frontier = &wl2;

  Timer t;
  t.Start();
  int iter = 0;
  int nitems = 1;
  insert<<<1, nthreads>>>(source, *in_frontier);
  nitems = in_frontier->nitems();
  do {
    ++ iter;
    nblocks = (nitems - 1) / nthreads + 1;
    printf("iteration %d: frontier_size = %d\n", iter, nitems);
    bellman_ford<<<nblocks, nthreads>>>(gg, d_dist, *in_frontier, *out_frontier);
    nitems = out_frontier->nitems();
    WLGPU *tmp = in_frontier;
    in_frontier = out_frontier;
    out_frontier = tmp;
    out_frontier->reset();
  } while (nitems > 0);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [sssp_gpu_base] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(ne) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
 
  CUDA_SAFE_CALL(cudaMemcpy(h_dist, d_dist, nv * sizeof(elabel_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_dist));
  return;
}

void BFSSolver(Graph &g, vidType source, vidType *dist) {}
