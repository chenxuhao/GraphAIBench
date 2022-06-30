// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"

vidType SampleFrequentElement(int m, vidType *comp, int64_t num_samples = 1024);

__device__ void link(vidType u, vidType v, comp_t *comp) {
  auto p1 = comp[u];
  auto p2 = comp[v];
  while (p1 != p2) {
    auto high = p1 > p2 ? p1 : p2;
    auto low = p1 + (p2 - high);
    auto p_high = comp[high];
    if ((p_high == low) || (p_high == high && 
          atomicCAS(&comp[high], high, low) == high))
      break;
    p1 = comp[comp[high]];
    p2 = comp[low];
  }
}

__global__ void compress(int m, comp_t *comp) {
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if(src < m) {
    while (comp[src] != comp[comp[src]]) {
      comp[src] = comp[comp[src]];
    }
  }
}

__global__ void afforest(GraphGPU g, comp_t *comp, int32_t r) {
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if(src < g.V()) {
    int row_begin = g.edge_begin(src);
    int row_end = g.edge_end(src); 
    int start_offset = min(r, row_end - row_begin);
    row_begin += start_offset;
    for (auto offset = row_begin; offset < row_end; offset ++) {
      auto dst = g.getEdgeDst(offset);
      link(src, dst, comp);
      break;
    }
  }
}

__global__ void afforest_undirected(GraphGPU g, int c, comp_t *comp, int r) {
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if(src < g.V() && comp[src] != c) {
    int row_begin = g.edge_begin(src);
    int row_end = g.edge_end(src); 
    int start_offset = min(r, row_end - row_begin);
    row_begin += start_offset;
    for (auto offset = row_begin; offset < row_end; offset ++) {
      auto dst = g.getEdgeDst(offset);
      link(src, dst, comp);
    }
  }
}

__global__ void afforest_directed(GraphGPU g, int c, comp_t *comp, int r) {
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if(src < g.V() && comp[src] != c) {
    int row_begin = g.edge_begin(src);
    int row_end = g.edge_end(src); 
    int start_offset = min(r, row_end - row_begin);
    row_begin += start_offset;
    for (auto offset = row_begin; offset < row_end; offset ++) {
      auto dst = g.getEdgeDst(offset);
      link(src, dst, comp);
    }
    row_begin = g.in_edge_begin(src);
    row_end = g.in_edge_end(src);
    for (auto offset = row_begin; offset < row_end; offset ++) {
      auto dst = g.getInEdgeDst(offset);
      link(src, dst, comp);
    }
  }
}

void CCSolver(Graph &g, comp_t *h_comp) {
  if (!g.has_reverse_graph()) {
    std::cout << "This algorithm requires the reverse graph constructed for directed graph\n";
    std::cout << "Please set reverse to 1 in the command line\n";
    exit(1);
  }
 
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
  int max_blocks_per_SM = maximum_residency(afforest, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA CC (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  comp_t *d_comp;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_comp, nv * sizeof(comp_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_comp, h_comp, nv * sizeof(comp_t), cudaMemcpyHostToDevice));

  Timer t;
  t.Start();
  int neighbor_rounds = 2;
  for (int r = 0; r < neighbor_rounds; ++r) {
    afforest<<<nblocks, nthreads>>>(gg, d_comp, r);
    compress<<<nblocks, nthreads>>>(nv, d_comp);
  }
  CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, nv * sizeof(comp_t), cudaMemcpyDeviceToHost));
  auto c = SampleFrequentElement(nv, h_comp);
  if (!g.is_directed()) {
    afforest_undirected<<<nblocks, nthreads>>>(gg, c, d_comp, neighbor_rounds);
  } else {
    afforest_directed<<<nblocks, nthreads>>>(gg, c, d_comp, neighbor_rounds);
  }
  compress<<<nblocks, nthreads>>>(nv, d_comp);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [cc_gpu_afforest] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(ne) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";

  CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, nv * sizeof(comp_t), cudaMemcpyDeviceToHost));
}

