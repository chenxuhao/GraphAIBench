// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"

__global__ void hook(GraphGPU g, comp_t *comp, bool *changed) {
  __shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
  const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

  for(int src = warp_id; src < g.V(); src += num_warps) {
    // use two threads to fetch row_offsets[src] and row_offsets[src+1]
    // this is considerably faster than the straightforward version
    if(thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(src + thread_lane);
    const int row_begin = ptrs[warp_lane][0];                   //same as: row_begin = row_offsets[isrc];
    const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = row_offsets[src+1];
    int comp_src = comp[src];
    for (int offset = row_begin + thread_lane; offset < row_end; offset += WARP_SIZE) {
      auto dst = g.getEdgeDst(offset);
      auto comp_dst = comp[dst];
      if (comp_src == comp_dst) continue;
      int high_comp = comp_src > comp_dst ? comp_src : comp_dst;
      int low_comp = comp_src + (comp_dst - high_comp);
      if (high_comp == comp[high_comp]) {
        *changed = true;
        comp[high_comp] = low_comp;
      }
    }
  }
}

__global__ void shortcut(int m, comp_t *comp) {
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if(src < m) {
    while (comp[src] != comp[comp[src]]) {
      comp[src] = comp[comp[src]];
    }
  }
}

void CCSolver(Graph &g, comp_t *h_comp) {
  size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  GraphGPU gg(g);
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (nv-1)/WARPS_PER_BLOCK+1;
  if (nblocks > 65536) nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(hook, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(max_blocks, nblocks);
  std::cout << "CUDA CC (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  comp_t *d_comp;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_comp, sizeof(comp_t) * nv));
  CUDA_SAFE_CALL(cudaMemcpy(d_comp, h_comp, nv * sizeof(comp_t), cudaMemcpyHostToDevice));
  bool h_changed, *d_changed;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));

  Timer t;
  t.Start();
  int iter = 0;
  do {
    ++ iter;
    h_changed = false;
    CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(h_changed), cudaMemcpyHostToDevice));
    //printf("iteration=%d\n", iter);
    hook<<<nblocks, nthreads>>>(gg, d_comp, d_changed);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    shortcut<<<(nv - 1) / nthreads + 1, nthreads>>>(nv, d_comp);
    CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
  } while (h_changed);
  t.Stop();

  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [cc_gpu_warp] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(ne) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";

  CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, sizeof(comp_t) * nv, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_changed));
}

