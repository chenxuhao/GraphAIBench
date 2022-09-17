// Copyright (c) 2022 MIT
// Author: Xuhao Chen
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "triangle_cta_compressed.cuh"

void triangle_count(Graph &g, uint64_t &total) {}

void triangle_count_compressed(Graph &g, uint64_t &total) {
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
  int max_blocks_per_SM = maximum_residency(cta_vertex_compressed, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  // allocate buffer for decompressed adjacency lists
  size_t per_block_buffer_size = 2 * size_t(md) * sizeof(vidType);
  size_t buffer_size = nblocks * per_block_buffer_size;
  std::cout << "buffer size: " << float(buffer_size)/float(1024*1024) << " MB\n";
  vidType *buffer;
  CUDA_SAFE_CALL(cudaMalloc((void **)&buffer, buffer_size));

  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
  cta_vertex_compressed<<<nblocks, nthreads>>>(gg, buffer, md, d_total);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime [tc_gpu_compressed] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(ne) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

