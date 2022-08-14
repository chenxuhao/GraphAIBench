// Copyright 2020 MIT
// Author: Xuhao Chen <cxh@mit.edu>
#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "utils.cuh"
#include "worklist.cuh"
#include "cuda_launch_config.hpp"
typedef Worklist2<vidType, vidType> WLGPU;

typedef cub::BlockScan<vidType, BLOCK_SIZE> BlockScan;

__device__ __forceinline__ void process_edge(GraphGPU g, int depth, eidType edge, vidType *depths, WLGPU &out_queue) {
  auto dst = g.getEdgeDst(edge);
  if (depths[dst] == MYINFINITY) {
    depths[dst] = depth;
    out_queue.push(dst);
  }
}

__device__ __forceinline__ void expandByCta(GraphGPU g, int depth, vidType *depths,
                                            WLGPU &in_queue, WLGPU &out_queue) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  vidType vertex;
  __shared__ int owner;
  __shared__ vidType sh_vertex;
  owner = -1;
  vidType size = 0;
  if (in_queue.pop_id(id, vertex)) {
    size = g.get_degree(vertex);
  }
  while (true) {
    if (size > BLOCK_SIZE)
      owner = threadIdx.x;
    __syncthreads();
    if (owner == -1) break;
    __syncthreads();
    if (owner == threadIdx.x) {
      sh_vertex = vertex;
      in_queue.invalidate(id);
      owner = -1;
      size = 0;
    }
    __syncthreads();
    auto row_begin = g.edge_begin(sh_vertex);
    auto row_end = g.edge_end(sh_vertex+1);
    auto neighbor_size = row_end - row_begin;
    auto num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
    for (auto i = threadIdx.x; i < num; i += blockDim.x) {
      auto edge = row_begin + i;
      //vidType dst = 0;
      //vidType ncnt = 0;
      if (i < neighbor_size) {
        process_edge(g, depth, edge, depths, out_queue);
        //dst = g.getEdgeDst(edge);
        //if(depths[dst] == MYINFINITY) {
        //  depths[dst] = depth;
        //  ncnt = 1;
        //}
      }
      //out_queue.push_1item<BlockScan>(ncnt, dst);
    }
  }
}

__device__ __forceinline__ void expandByWarp(GraphGPU g, int depth, vidType *depths,
                                             WLGPU &in_queue, WLGPU &out_queue) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = threadIdx.x >> LOG_WARP_SIZE;
  int lane_id = LaneId();
  __shared__ int owner[NUM_WARPS];
  __shared__ vidType sh_vertex[NUM_WARPS];
  owner[warp_id] = -1;
  vidType size = 0;
  vidType vertex;
  if (in_queue.pop_id(id, vertex)) {
    if (vertex != vidType(-1))
      size = g.get_degree(vertex);
  }
  while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
    if(size >= WARP_SIZE)
      owner[warp_id] = lane_id;
    if(owner[warp_id] == lane_id) {
      sh_vertex[warp_id] = vertex;
      in_queue.invalidate(id);
      owner[warp_id] = -1;
      size = 0;
    }
    auto winner = sh_vertex[warp_id];
    auto row_begin = g.edge_begin(winner);
    auto row_end = g.edge_end(winner);
    auto neighbor_size = row_end - row_begin;
    auto num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    for(auto i = lane_id; i < num; i+= WARP_SIZE) {
      auto edge = row_begin + i;
      if(i < neighbor_size) {
        process_edge(g, depth, edge, depths, out_queue);
      }
    }
  }
}

__global__ void bfs_step(GraphGPU g, int depth, vidType *depths, 
                         WLGPU in_queue, WLGPU out_queue) {
  //expandByCta(g, depth, depths, in_queue, out_queue);
  //expandByWarp(g, depth, depths, in_queue, out_queue);
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  vidType vertex;
  const int SCRATCHSIZE = BLOCK_SIZE;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ eidType gather_offsets[SCRATCHSIZE];
  gather_offsets[threadIdx.x] = 0;
  vidType neighbor_size = 0;
  eidType neighbor_offset = 0;
  vidType scratch_offset = 0;
  vidType total_edges = 0;
  if (in_queue.pop_id(id, vertex)) {
    if (vertex != vidType(-1)) {
      neighbor_offset = g.edge_begin(vertex);
      neighbor_size = g.get_degree(vertex);
    }
  }
  BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
  int done = 0;
  int neighbors_done = 0;
  while (total_edges > 0) {
    __syncthreads();
    int i;
    for (i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
      gather_offsets[scratch_offset + i - done] = neighbor_offset + neighbors_done + i;
    }
    neighbors_done += i;
    scratch_offset += i;
    __syncthreads();
    auto edge = gather_offsets[threadIdx.x];
    if (threadIdx.x < total_edges) {
      process_edge(g, depth, edge, depths, out_queue);
    }
    total_edges -= BLOCK_SIZE;
    done += BLOCK_SIZE;
  }
}

__global__ void insert(vidType source, WLGPU in_queue) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id == 0) in_queue.push(source);
  return;
}

void BFSSolver(Graph &g, vidType source, vidType *h_depths) {
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
  vidType * d_depths;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, nv * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_depths, h_depths, nv * sizeof(vidType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(&d_depths[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  WLGPU queue1(ne), queue2(ne);
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
    nblocks = (nitems + nthreads - 1) / nthreads; 
    printf("iteration=%d, frontier_size=%d\n", iter, nitems);
    bfs_step<<<nblocks, nthreads>>>(gg, iter, d_depths, *in_frontier, *out_frontier);
    nitems = out_frontier->nitems();
    WLGPU *tmp = in_frontier;
    in_frontier = out_frontier;
    out_frontier = tmp;
    out_frontier->reset();
  } while(nitems > 0);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [gpu_twc] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(ne) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";

  CUDA_SAFE_CALL(cudaMemcpy(h_depths, d_depths, nv * sizeof(vidType), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_depths));
  return;
}

void SSSPSolver(Graph &g, vidType source, elabel_t *dist, int delta) {}
