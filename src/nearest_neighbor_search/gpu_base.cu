// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include "ann.h"
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

void ANN(Graph &g, std::vector<Embedding> points, Embedding query) {
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  int dim = query.size();
  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (g.V()-1)/WARPS_PER_BLOCK+1;
  std::cout << "CUDA ANN (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  float* h_query = query.data();
  float* d_query;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_query, sizeof(float)*dim));
  CUDA_SAFE_CALL(cudaMemcpy(d_query, &h_query, sizeof(float)*dim, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  cudaProfilerStart();
  Timer t;
  t.Start();
  // add code here
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  cudaProfilerStop();
  std::cout << "runtime [gpu_base] = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaFree(d_query));
}

