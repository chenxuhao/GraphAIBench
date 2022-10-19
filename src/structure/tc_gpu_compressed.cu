// Copyright (c) 2022 MIT
// Author: Xuhao Chen
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
//#define TC_BS_CTA_VERTEX_COMPRESSED
#define TC_HINDEX_WARP_VERTEX_COMPRESSED
#ifdef TC_BS_CTA_VERTEX_COMPRESSED
#include "triangle_cta_compressed.cuh"
#else
#ifdef TC_HINDEX_WARP_VERTEX_COMPRESSED
#include "triangle_hindex_warp_vertex_compressed.cuh"
#endif
#endif

void triangle_count(Graph &g, uint64_t &total) {}

void triangle_count_compressed(Graph &g, uint64_t &total) {
  size_t memsize = print_device_info(0);
  GraphGPU gg(g);

  // kernel launch configuration
  size_t nthreads = BLOCK_SIZE, nblocks = (g.V()-1)/nthreads+1;
  if (nblocks > 65536) nblocks = 65536;
#ifdef TC_BS_CTA_VERTEX_COMPRESSED
  //refine_kernel_config(nthreads, nblocks, cta_vertex_compressed);
#else
  refine_kernel_config(nthreads, nblocks, hindex_warp_vertex_compressed);
#endif
  std::cout << "CUDA kernel (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  std::cout << "Allocating buffer for decompressed adjacency lists\n";
  vidType *buffer;
  size_t num_per_block = WARPS_PER_BLOCK;
#ifdef TC_BS_CTA_VERTEX_COMPRESSED
  num_per_block = 1;
#endif
  allocate_gpu_buffer(3 * size_t(g.get_max_degree()) * num_per_block * nblocks, buffer);

#ifdef TC_HINDEX_WARP_VERTEX_COMPRESSED
  std::cout << "Allocating buckets for the hash map\n";
  vidType *bins;
  allocate_gpu_buffer(NUM_BUCKETS * BUCKET_SIZE * WARPS_PER_BLOCK * nblocks, bins);
  auto bins_mem = NUM_BUCKETS * BUCKET_SIZE * WARPS_PER_BLOCK * nblocks * sizeof(vidType);
  CUDA_SAFE_CALL(cudaMemset(bins, 0, bins_mem));
#endif

  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
#ifdef TC_BS_CTA_VERTEX_COMPRESSED
  cta_vertex_compressed<<<nblocks, nthreads>>>(gg, buffer, g.get_max_degree(), d_total);
#else
#ifdef TC_HINDEX_WARP_VERTEX_COMPRESSED
  std::cout << "Use HINDEX\n";
  hindex_warp_vertex_compressed<<<nblocks, nthreads>>>(gg, bins, buffer, g.get_max_degree(), d_total);
  //hindex_warp_vertex<<<nblocks, nthreads>>>(gg, bins, d_total);
#endif
#endif
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(g.E()) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
  CUDA_SAFE_CALL(cudaFree(buffer));
#ifdef TC_HINDEX_WARP_VERTEX_COMPRESSED
  CUDA_SAFE_CALL(cudaFree(bins));
#endif
}

