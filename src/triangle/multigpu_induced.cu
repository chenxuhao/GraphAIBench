// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
//#include "edgelist.h"
#include "graph_gpu.h"
#include "graph_partition.h"
#include "scheduler.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "bs_warp_edge.cuh"
#include "bs_warp_vertex.cuh"
#include <thread>

void TCSolver(Graph &g, uint64_t &total, int n_gpus, int chunk_size) {
  int ndevices = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";
  //eidType nnz = g.init_edgelist();
  //std::cout << "Total edgelist size = " << nnz << "\n";

  if (ndevices < n_gpus) {
    std::cout << "Only " << ndevices << " GPUs available\n";
  } else ndevices = n_gpus;
  PartitionedGraph pg(&g, ndevices);
  pg.edgecut_induced_partition1D();
 
  std::vector<GraphGPU> d_graphs(ndevices);
  Timer t;
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    d_graphs[i].init(*pg.get_subgraph(i), i, ndevices);
    //d_graphs[i].copy_edgelist_to_device(num_tasks, src_ptrs, dst_ptrs);
  }
  t.Stop();
  std::cout << "Total GPU copy time (graph+edgelist) = " << t.Seconds() <<  " sec\n";

  size_t nthreads = BLOCK_SIZE;
  std::vector<AccType> h_counts(ndevices, 0);
  size_t nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(warp_vertex, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(6*max_blocks, nblocks); 
  std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  std::vector<AccType *> d_count(ndevices);
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMalloc(&d_count[i], sizeof(AccType)));
  }
  std::vector<std::thread> threads;
  std::vector<Timer> subt(ndevices);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
 
  t.Start();
  for (int i = 0; i < ndevices; i++) {
    threads.push_back(std::thread([&,i]() {
    cudaSetDevice(i);
    subt[i].Start();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(d_count[i], &h_counts[i], sizeof(AccType), cudaMemcpyHostToDevice));
    //cudaMemcpyAsync(d_count[i], &h_counts[i], sizeof(AccType), cudaMemcpyHostToDevice);
    auto begin = pg.get_local_begin(i);
    auto end = pg.get_local_end(i);
    warp_vertex<<<nblocks, nthreads>>>(begin, end, d_graphs[i], d_count[i]);
    CUDA_SAFE_CALL(cudaMemcpy(&h_counts[i], d_count[i], sizeof(AccType), cudaMemcpyDeviceToHost));
    //cudaMemcpyAsync(&h_counts[i], d_count[i], sizeof(AccType), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    subt[i].Stop();
    }));
  }
  for (auto &thread: threads) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  for (int i = 0; i < ndevices; i++) total += h_counts[i];
  t.Stop();
  for (int i = 0; i < ndevices; i++)
    std::cout << "runtime[gpu" << i << "] = " << subt[i].Seconds() <<  " sec\n";
  std::cout << "runtime = " << t.Seconds() <<  " sec\n";
}

