// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

#define USE_ZERO_COPY 0

#ifdef VERTEX_PAR
#ifdef CTA_CENTRIC
const std::string name = "gpu_cta_vp";
#include "bs_cta_vertex.cuh"
#else
const std::string name = "gpu_warp_vp";
#include "bs_warp_vertex.cuh"
#endif
#else
#ifdef CTA_CENTRIC
const std::string name = "gpu_cta_ep";
#include "bs_cta_edge.cuh"
#else
const std::string name = "gpu_warp_ep";
#include "bs_warp_edge.cuh"
#endif
#endif

void TCSolver(Graph &g, uint64_t &total, int, int) {
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();

  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
#ifdef VERTEX_PAR
  size_t nblocks = (g.V()-1)/WARPS_PER_BLOCK+1;
#else
  auto nnz = gg.init_edgelist(g, 0, 0, USE_ZERO_COPY); // streaming edgelist using zero-copy
  size_t nblocks = (nnz-1)/WARPS_PER_BLOCK+1;
#endif
  if (nblocks > 65536) nblocks = 65536;

  std::cout << "Using BinarySearch ";
#ifdef VERTEX_PAR
  std::cout << "Vertex-parallel ";
  std::cout << "Warp-centric\n";
  refine_kernel_config(nthreads, nblocks, triangle_bs_warp_vertex);
#else
  std::cout << "Edge-parallel ";
  if (nblocks > 65536) nblocks = 65536;
#ifdef CTA_CENTRIC
  std::cout << "CTA-centric\n";
  refine_kernel_config(nthreads, nblocks, triangle_bs_cta_edge);
#else
  std::cout << "Warp-centric\n";
  refine_kernel_config(nthreads, nblocks, triangle_bs_warp_edge);
#endif
#endif
  std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  AccType h_total = 0, *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  cudaProfilerStart();
  Timer t;
  t.Start();
#ifdef VERTEX_PAR
#ifdef CTA_CENTRIC
  triangle_bs_cta_vertex<<<nblocks, nthreads>>>(0, g.V(), gg, d_total);
#else
  triangle_bs_warp_vertex<<<nblocks, nthreads>>>(0, g.V(), gg, d_total);
#endif
#else
#ifdef CTA_CENTRIC
  triangle_bs_cta_edge<<<nblocks, nthreads>>>(gg, d_total);
#else
  triangle_bs_warp_edge<<<nblocks, nthreads>>>(gg, d_total);
#endif
#endif
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  cudaProfilerStop();

  std::cout << "runtime [" << name << "] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(g.E()) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
}

