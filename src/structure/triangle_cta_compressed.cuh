// CTA-centric vertex parallel: each thread block takes one vertex
__global__ void cta_vertex_compressed(GraphGPU g, vidType *buffer, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  AccType count = 0;
  vidType *adj_list = buffer + max_deg*blockIdx.x;
  for (vidType v = blockIdx.x; v < g.V(); v += gridDim.x) {
    auto deg = g.cta_decompress(v, adj_list);
    for (vidType i = 0; i < deg; i++) {
      auto u = adj_list[i];
      count += g.cta_intersect_compressed(v, u);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

