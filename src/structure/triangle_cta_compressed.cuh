// CTA-centric edge parallel: each thread block takes one edge
__global__ void cta_triangle_compressed(GraphGPU g, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  AccType count = 0;
  for (vidType v = blockIdx.x; v < g.V(); v += gridDim.x) {
    for (vidType i = 0; i < g.get_degree_compressed(u); i++) {
      auto u = g.N_compressed(v, i);
      count += g.cta_intersect_compressed(v, u);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

