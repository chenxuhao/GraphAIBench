// Copyright (c) 2022 MIT
#include <cub/cub.cuh>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "graph_gpu.h"
#include "utils.cuh"
#include "worklist.cuh"
#include "cuda_launch_config.hpp"

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
typedef cub::BlockReduce<score_t, BLOCK_SIZE> BlockReduce;

__global__ void initialize(int m, int *depths) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < m) depths[id] = -1;
}

// Shortest path calculation by forward BFS
__global__ void forward_base(GraphGPU g, int depth, int *path_counts, int *depths, 
                             Worklist2 in_queue, Worklist2 out_queue) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int src;
  if (in_queue.pop_id(tid, src)) {
    auto row_begin = g.edge_begin(src);
    auto row_end = g.edge_end(src); 
    for (auto offset = row_begin; offset < row_end; ++ offset) {
      auto dst = g.getEdgeDst(offset);
      if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1)) {
        assert(out_queue.push(dst));
      }
      if (depths[dst] == depth) {
        atomicAdd(&path_counts[dst], path_counts[src]);
      }
    }
  }
}

// Dependency accumulation by back propagation
__global__ void reverse_base(int num, GraphGPU g, int depth, int start, 
                             const vidType* frontiers, score_t *scores, 
                             const int *path_counts, int *depths, score_t *deltas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int src = frontiers[start + tid];
    auto row_begin = g.edge_begin(src);
    auto row_end = g.edge_end(src); 
    for (auto offset = row_begin; offset < row_end; ++ offset) {
      auto dst = g.getEdgeDst(offset);
      if (depths[dst] == depth + 1) {
        deltas[src] += static_cast<score_t>(path_counts[src]) / 
          static_cast<score_t>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
      }
    }
    scores[src] += deltas[src];
  }
}

__device__ __forceinline__ void process_edge(GraphGPU g, int value, int depth, int offset, 
                                             int *path_counts, int *depths, Worklist2 &out_queue) {
  int dst = g.getEdgeDst(offset);
  if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1)) {
    assert(out_queue.push(dst));
  }
  if (depths[dst] == depth) atomicAdd(&path_counts[dst], value);
}

__device__ __forceinline__ void expandByCta(GraphGPU g, int depth,
                                            int *path_counts, int *depths, 
                                            Worklist2 &in_queue, Worklist2 &out_queue) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int owner;
  __shared__ int sh_src;
  owner = -1;
  int size = 0;
  int src;
  if(in_queue.pop_id(id, src)) {
    size = g.get_degree(src);
  }
  while(true) {
    if(size > BLOCK_SIZE)
      owner = threadIdx.x;
    __syncthreads();
    if(owner == -1) break;
    __syncthreads();
    if(owner == threadIdx.x) {
      sh_src = src;
      in_queue.d_queue[id] = -1;
      owner = -1;
      size = 0;
    }
    __syncthreads();
    auto row_begin = g.edge_begin(sh_src);
    auto row_end = g.edge_end(sh_src);
    auto neighbor_size = row_end - row_begin;
    int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
    int value = path_counts[sh_src];
    for(int i = threadIdx.x; i < num; i += blockDim.x) {
      int dst = 0;
      int ncnt = 0;
      int offset = row_begin + i;
      if(i < neighbor_size) {
        dst = g.getEdgeDst(offset);
        if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth) == -1))
          ncnt = 1;
        if (depths[dst] == depth) atomicAdd(&path_counts[dst], value);
      }
      out_queue.push_1item<BlockScan>(ncnt, dst, BLOCK_SIZE);
    }
  }
}

__device__ __forceinline__ void expandByWarp(GraphGPU g, int depth,
                                             int *path_counts, int *depths, 
                                             Worklist2 &in_queue, Worklist2 &out_queue) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
  unsigned lane_id = LaneId();
  __shared__ int owner[NUM_WARPS];
  __shared__ int sh_src[NUM_WARPS];
  owner[warp_id] = -1;
  int size = 0;
  int src;
  if(in_queue.pop_id(id, src)) {
    if (src != -1) size = g.get_degree(src);
  }
  while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
    if(size >= WARP_SIZE)
      owner[warp_id] = lane_id;
    if(owner[warp_id] == lane_id) {
      sh_src[warp_id] = src;
      in_queue.d_queue[id] = -1;
      owner[warp_id] = -1;
      size = 0;
    }
    int winner = sh_src[warp_id];
    int row_begin = g.edge_begin(winner);
    int row_end = g.edge_end(winner);
    int neighbor_size = row_end - row_begin;
    int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int value = path_counts[winner];
    for(int i = lane_id; i < num; i+= WARP_SIZE) {
      int edge = row_begin + i;
      if(i < neighbor_size) {
        process_edge(g, value, depth, edge, path_counts, depths, out_queue);
      }
    }
  }
}

__global__ void forward_lb(GraphGPU g, int depth, int *path_counts, int *depths, 
                           Worklist2 in_queue, Worklist2 out_queue) {
  //expandByCta(g, depth, path_counts, depths, in_queue, out_queue);
  //expandByWarp(g, depth, path_counts, depths, in_queue, out_queue);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;
  int src;
  const int SCRATCHSIZE = BLOCK_SIZE;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ int gather_offsets[SCRATCHSIZE];
  __shared__ int srcsrc[SCRATCHSIZE];
  __shared__ int values[BLOCK_SIZE];
  gather_offsets[threadIdx.x] = 0;
  int neighbor_size = 0;
  int neighbor_offset = 0;
  int scratch_offset = 0;
  int total_edges = 0;
  if(in_queue.pop_id(tid, src)) {
    if(src != -1) {
      neighbor_offset = g.edge_begin(src);
      neighbor_size = g.get_degree(src);
      values[tx] = path_counts[src];
    }
  }
  BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
  int done = 0;
  int neighbors_done = 0;
  while(total_edges > 0) {
    __syncthreads();
    int i;
    for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
      gather_offsets[scratch_offset + i - done] = neighbor_offset + neighbors_done + i;
      srcsrc[scratch_offset + i - done] = tx;
    }
    neighbors_done += i;
    scratch_offset += i;
    __syncthreads();
    if(tx < total_edges) {
      int edge = gather_offsets[tx];
      process_edge(g, values[srcsrc[tx]], depth, edge, path_counts, depths, out_queue);
    }
    total_edges -= BLOCK_SIZE;
    done += BLOCK_SIZE;
  }
}

__global__ void bc_reverse_warp(int num, GraphGPU g, int depth, int start, 
                                const vidType *frontiers, score_t *scores, 
                                const int *path_counts, int *depths, score_t *deltas) {
  __shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
  __shared__ score_t sdata[BLOCK_SIZE + 16];                       // padded to avoid reduction conditionals

  const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for(int index = warp_id; index < num; index += num_warps) {
    int src = frontiers[start + index];
    // use two threads to fetch Ap[row] and Ap[row+1]
    // this is considerably faster than the straightforward version
    if(thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(src + thread_lane);
    const int row_begin = ptrs[warp_lane][0];   //same as: row_start = row_offsets[row];
    const int row_end   = ptrs[warp_lane][1];   //same as: row_end   = row_offsets[row+1];
    score_t sum = 0;
    for(int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
      int dst = g.getEdgeDst(offset);
      if(depths[dst] == depth + 1) {
        sum += static_cast<score_t>(path_counts[src]) / 
          static_cast<score_t>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
      }
    }
    // store local sum in shared memory
    sdata[threadIdx.x] = sum; __syncthreads();

    // reduce local sums to row sum
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
    if (thread_lane == 0) {
      deltas[src] += sdata[threadIdx.x];
      scores[src] += deltas[src];
    }
  }
}

__device__ __forceinline__ void reverse_expand_cta(int num, GraphGPU g,
                                                   int start, int depth, IndexT *frontiers, 
                                                   score_t *scores, const int *path_counts, 
                                                   int *depths, score_t *deltas) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int src = 0;
  int size = 0;
  __shared__ int owner;
  __shared__ int sh_src;
  owner = -1;
  if(tid < num) {
    src = frontiers[start + tid];
    size = g.get_degree(src);
  }
  while(true) {
    if(size > BLOCK_SIZE)
      owner = threadIdx.x;
    __syncthreads();
    if(owner == -1) break;
    __syncthreads();
    if(owner == threadIdx.x) {
      sh_src = src;
      frontiers[start + tid] = -1;
      owner = -1;
      size = 0;
    }
    __syncthreads();
    int row_begin = g.edge_begin(sh_src);
    int row_end = g.edge_end(sh_src);
    int neighbor_size = row_end - row_begin;
    int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
    int count = path_counts[sh_src];
    score_t sum = 0;
    for(int i = threadIdx.x; i < num; i += blockDim.x) {
      int offset = row_begin + i;
      if(i < neighbor_size) {
        int dst = g.getEdgeDst(offset);
        if(depths[dst] == depth + 1) {
          score_t value = static_cast<score_t>(count) /
            static_cast<score_t>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
          sum += value;
        }
      }
    }
    score_t delta_src = BlockReduce(temp_storage).Sum(sum);
    if(threadIdx.x == 0) {
      deltas[sh_src]  = delta_src;
      scores[sh_src] += delta_src;
    }
  }
}

__device__ __forceinline__ void reverse_expand_warp(int num, GraphGPU g, int depth, int start, 
                                                    vidType *frontiers, score_t *scores, 
                                                    const int *path_counts, int *depths, score_t *deltas) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
  unsigned lane_id = LaneId();
  __shared__ int owner[NUM_WARPS];
  __shared__ int sh_src[NUM_WARPS];
  __shared__ score_t sdata[BLOCK_SIZE + 16];
  owner[warp_id] = -1;
  int size = 0;
  int src = -1;
  if(tid < num) {
    src = frontiers[start + tid];
    if(src != -1) {
      size = g.get_degree(src);
    }
  }
  while(__any_sync(0xFFFFFFFF, size) >= WARP_SIZE) {
    if(size >= WARP_SIZE)
      owner[warp_id] = lane_id;
    if(owner[warp_id] == lane_id) {
      sh_src[warp_id] = src;
      frontiers[start + tid] = -1;
      owner[warp_id] = -1;
      size = 0;
    }
    int winner = sh_src[warp_id];
    int row_begin = g.edge_begin(winner);
    int row_end = g.edge_end(winner);
    int neighbor_size = row_end - row_begin;
    int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int count = path_counts[winner];
    score_t sum = 0;
    for(int i = lane_id; i < num; i+= WARP_SIZE) {
      int edge = row_begin + i;
      if(i < neighbor_size) {
        int dst = g.getEdgeDst(edge);
        if(depths[dst] == depth + 1) {
          score_t value = static_cast<score_t>(count) /
            static_cast<score_t>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
          sum += value;
        }
      }
    }
    sdata[threadIdx.x] = sum; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
    sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
    if(lane_id == 0) {
      deltas[winner]  = sdata[threadIdx.x];
      scores[winner] += sdata[threadIdx.x];
    }
  }
}

__global__ void reverse_lb(int num, GraphGPU g, int depth, int start, 
                           vidType *frontiers, score_t *scores, 
                           const int *path_counts, int *depths, score_t *deltas) {
  //reverse_expand_cta(num, g, depth, start, frontiers, scores, path_counts, depths, deltas);
  //reverse_expand_warp(num, g, depth, start, frontiers, scores, path_counts, depths, deltas);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ int gather_offsets[BLOCK_SIZE];
  //__shared__ int srcs[BLOCK_SIZE];
  __shared__ int idx[BLOCK_SIZE];
  __shared__ int sh_counts[BLOCK_SIZE];
  __shared__ score_t sh_deltas[BLOCK_SIZE];
  gather_offsets[tx] = 0;
  //srcs[tx] = 0;
  idx[tx] = 0;
  sh_counts[tx] = 0;
  sh_deltas[tx] = 0;
  int neighbor_size = 0;
  int neighbor_offset = 0;
  int scratch_offset = 0;
  int total_edges = 0;
  int src = -1;
  if(tid < num) {
    src = frontiers[start + tid];
    if(src != -1) {
      neighbor_offset = g.edge_begin(src);
      neighbor_size = g.get_degree(src);
      sh_counts[tx] = path_counts[src];
    }
  }
  BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
  int done = 0;
  int neighbors_done = 0;
  while(total_edges > 0) {
    __syncthreads();
    int i;
    for(i = 0; neighbors_done + i < neighbor_size && (scratch_offset + i - done) < BLOCK_SIZE; i++) {
      int j = scratch_offset + i - done;
      gather_offsets[j] = neighbor_offset + neighbors_done + i;
      //srcs[j] = src;
      idx[j] = tx;
    }
    neighbors_done += i;
    scratch_offset += i;
    __syncthreads();
    if(tx < total_edges) {
      int offset = gather_offsets[tx];
      int dst = g.getEdgeDst(offset);
      if(depths[dst] == depth + 1) {
        score_t value = static_cast<score_t>(sh_counts[idx[tx]]) / 
          //score_t value = static_cast<score_t>(path_counts[srcs[tx]]) / 
          static_cast<score_t>(__ldg(path_counts+dst)) * (1 + deltas[dst]);
        atomicAdd(&sh_deltas[idx[tx]], value); 
      }
    }
    total_edges -= BLOCK_SIZE;
    done += BLOCK_SIZE;
  }
  __syncthreads();
  if(src != -1) {
    deltas[src]  = sh_deltas[tx];
    scores[src] += sh_deltas[tx];
  }
}

__global__ void insert(Worklist2 in_queue, int src, int *path_counts, int *depths) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id == 0) {
    in_queue.push(src);
    path_counts[src] = 1;
    depths[src] = 0;
  }
  return;
}

__global__ void push_frontier(Worklist2 in_queue, int *queue, int queue_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int vertex;
  if (in_queue.pop_id(tid, vertex)) {
    queue[queue_len+tid] = vertex;
  }
}

__global__ void bc_normalize(int m, score_t *scores, score_t max_score) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < m) scores[tid] = scores[tid] / (max_score);
}

void BCSolver(Graph &g, int source, score_t *h_scores) {
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (nv-1)/nthreads+1;
  //if (nblocks > 65536) nblocks = 65536;
  assert(nblocks < 65536);
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(forward_base, nthreads, 0);
  //max_blocks_per_SM = maximum_residency(reverse_base, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA Betweenness Centrality (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  score_t *d_scores, *d_deltas;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(score_t) * nv));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, sizeof(score_t) * nv));
  CUDA_SAFE_CALL(cudaMemset(d_scores, 0, nv * sizeof(score_t)));
  CUDA_SAFE_CALL(cudaMemset(d_deltas, 0, nv * sizeof(score_t)));
  int *d_path_counts, *d_depths, *d_frontiers;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_path_counts, sizeof(int) * nv));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, sizeof(int) * nv));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontiers, sizeof(int) * (nv+1)));
  CUDA_SAFE_CALL(cudaMemset(d_path_counts, 0, nv * sizeof(int)));

  int depth = 0;
  int nitems = 1;
  int frontiers_len = 0;
  vector<int> depth_index;
  depth_index.push_back(0);
  Worklist2 wl1(nv), wl2(nv);
  Worklist2 *inwl = &wl1, *outwl = &wl2;
  initialize <<<nblocks, nthreads>>> (nv, d_depths);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  Timer t;
  t.Start();
  insert<<<1, 1>>>(*inwl, source, d_path_counts, d_depths);
  do {
    nblocks = (nitems - 1) / nthreads + 1;
    push_frontier<<<nblocks, nthreads>>>(*inwl, d_frontiers, frontiers_len);
    frontiers_len += nitems;
    depth_index.push_back(frontiers_len);
    printf("Forward: depth=%d, frontire_size=%d\n", depth, nitems);
    depth++;
    forward_lb<<<nblocks, nthreads>>>(gg, depth, d_path_counts, d_depths, *inwl, *outwl);
    //forward_base<<<nblocks, nthreads>>>(gg, depth, d_path_counts, d_depths, *inwl, *outwl);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nitems = outwl->nitems();
    Worklist2 *tmp = inwl;
    inwl = outwl;
    outwl = tmp;
    outwl->reset();
  } while (nitems > 0);
  for (int d = depth_index.size() - 2; d >= 0; d--) {
    nitems = depth_index[d+1] - depth_index[d];
    nblocks = (nitems - 1) / nthreads + 1;
    printf("Reverse: depth=%d, frontier_size=%d\n", d, nitems);
#ifdef REVERSE_WARP
    nblocks = std::min(max_blocks, DIVIDE_INTO(nitems, WARPS_PER_BLOCK));
    bc_reverse_warp<<<nblocks, nthreads>>>(nitems, gg, d, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d_deltas);
#else
    reverse_lb<<<nblocks, nthreads>>>(nitems, gg, d, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d_deltas);
    //reverse_base<<<nblocks, nthreads>>>(nitems, gg, d, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d_deltas);
#endif
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }
  score_t *d_max_score;
  d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + nv);
  score_t h_max_score;
  CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(score_t), cudaMemcpyDeviceToHost));
  nthreads = 512;
  nblocks = (nv - 1) / nthreads + 1;
  bc_normalize<<<nblocks, nthreads>>>(nv, d_scores, h_max_score);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  std::cout << "max_score = " << h_max_score << "\n";
  std::cout << "iterations = " << depth << "\n";
  std::cout << "runtime [bc_gpu_twc] = " << t.Seconds() << " sec\n";
 
  CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(score_t) * nv, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_path_counts));
  CUDA_SAFE_CALL(cudaFree(d_depths));
  CUDA_SAFE_CALL(cudaFree(d_deltas));
  CUDA_SAFE_CALL(cudaFree(d_frontiers));
}

