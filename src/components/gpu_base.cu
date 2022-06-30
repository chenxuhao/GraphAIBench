// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"

__global__ void hook(GraphGPU g, comp_t *comp, bool *changed) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if(src < g.V()) {
		auto comp_src = comp[src];
		auto row_begin = g.edge_begin(src);
		auto row_end = g.edge_end(src); 
		for (auto offset = row_begin; offset < row_end; ++ offset) {
			auto dst = g.getEdgeDst(offset);
			auto comp_dst = comp[dst];
			if (comp_src == comp_dst) continue;
			auto high_comp = comp_src > comp_dst ? comp_src : comp_dst;
			auto low_comp = comp_src + (comp_dst - high_comp);
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
  size_t nblocks = (nv-1)/nthreads+1;
  assert(nblocks < 65536);
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(hook, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  //size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  //nblocks = std::min(max_blocks, nblocks);
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
    shortcut<<<nblocks, nthreads>>>(nv, d_comp);
    CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
  } while (h_changed);
  t.Stop();

  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [cc_gpu_base] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(ne) / t.Seconds() / 1e9 << " billion Traversed Edges Per Second (TEPS)\n";
 
  CUDA_SAFE_CALL(cudaMemcpy(h_comp, d_comp, sizeof(comp_t) * nv, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_changed));
}

