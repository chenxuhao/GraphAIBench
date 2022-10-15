// Copyright (c) 2022 MIT
// Author: Xuhao Chen
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
//#define TC_CTA_BS_COMPRESSED
#ifdef TC_CTA_BS_COMPRESSED
#include "triangle_cta_compressed.cuh"
#else
//#if TC_WARP_HINDEX_COMPRESSED
#include "triangle_hindex_warp_vertex_compressed.cuh"
#endif

void triangle_count(Graph &g, uint64_t &total) {}

void triangle_count_compressed(Graph &g, uint64_t &total) {
  size_t memsize = print_device_info(0);
  GraphGPU gg(g);

  // kernel launch configuration
  size_t nthreads = BLOCK_SIZE, nblocks = (g.V()-1)/nthreads+1;
  if (nblocks > 65536) nblocks = 65536;
#ifdef TC_CTA_BS_COMPRESSED
  //refine_kernel_config(nthreads, nblocks, cta_vertex_compressed);
#else
  //refine_kernel_config(nthreads, nblocks, hindex_warp_vertex_compressed);
#endif
  std::cout << "CUDA kernel (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  // allocate buffer for decompressed adjacency lists
  vidType *buffer;
  allocate_gpu_vertex_buffer(3 * size_t(g.get_max_degree()), nblocks, buffer);

#ifndef TC_CTA_BS_COMPRESSED
  vidType *bins;
  allocate_gpu_vertex_buffer(NUM_BUCKETS * BUCKET_SIZE, WARPS_PER_BLOCK*nblocks, bins);
  auto bins_mem = NUM_BUCKETS * BUCKET_SIZE * WARPS_PER_BLOCK * nblocks * sizeof(vidType);
  CUDA_SAFE_CALL(cudaMemset(bins, 0, bins_mem));
#endif

  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
#ifdef TC_CTA_BS_COMPRESSED
  cta_vertex_compressed<<<nblocks, nthreads>>>(gg, buffer, g.get_max_degree(), d_total);
#else
  hindex_warp_vertex_compressed<<<nblocks, nthreads>>>(gg, bins, buffer, g.get_max_degree(), d_total);
#endif
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "runtime = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(g.E()) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

