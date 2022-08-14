// Copyright 2020 MIT
// Author: Xuhao Chen <cxh@mit.edu>
#include "graph_gpu.h"
#include "worklist.cuh"
#include "cuda_launch_config.hpp"
typedef Worklist2<vidType, vidType> WLGPU;

__global__ void bfs_step(GraphGPU g, vidType *dists, WLGPU in_queue, WLGPU out_queue) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  vidType src;
  if (in_queue.pop_id(tid, src)) {
    auto row_begin = g.edge_begin(src);
    auto row_end = g.edge_end(src);
    for (auto offset = row_begin; offset < row_end; ++ offset) {
      auto dst = g.getEdgeDst(offset);
      if ((dists[dst] == MYINFINITY) && 
          (atomicCAS(&dists[dst], MYINFINITY, dists[src]+1) == MYINFINITY)) {
        assert(out_queue.push(dst));
      }
    }
  }
}

__global__ void insert(vidType source, WLGPU queue) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id == 0) queue.push(source);
  return;
}

void BFSSolver(Graph &g, vidType source, vidType *h_dists) {
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
  auto max_blocks_per_SM = maximum_residency(bfs_step, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA BFS (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  vidType zero = 0;
  vidType * d_dists;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_dists, nv * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_dists, h_dists, nv * sizeof(vidType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(&d_dists[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  WLGPU queue1(nv), queue2(nv);
  WLGPU *in_frontier = &queue1, *out_frontier = &queue2;
  std::cout << "CUDA BFS (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

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
    bfs_step<<<nblocks, nthreads>>>(gg, d_dists, *in_frontier, *out_frontier);
    nitems = out_frontier->nitems();
    WLGPU *tmp = in_frontier;
    in_frontier = out_frontier;
    out_frontier = tmp;
    out_frontier->reset();
  } while (nitems > 0);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [gpu_base] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(ne) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";

  CUDA_SAFE_CALL(cudaMemcpy(h_dists, d_dists, nv * sizeof(vidType), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_dists));
  return;
}

void SSSPSolver(Graph &g, vidType source, elabel_t *dist, int delta) {}
