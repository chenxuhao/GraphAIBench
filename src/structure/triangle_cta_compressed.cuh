// CTA-centric vertex parallel: each thread block takes one vertex
__global__ void cta_vertex_compressed(GraphGPU g, vidType *buffer, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  AccType count = 0;
  vidType *adj_v = buffer + max_deg*(2*blockIdx.x);
  vidType *adj_u = buffer + max_deg*(2*blockIdx.x+1);
  for (vidType v = blockIdx.x; v < g.V(); v += gridDim.x) {
    auto deg = g.cta_decompress(v, adj_v);
    for (vidType i = 0; i < deg; i++) {
      auto u = adj_v[i];
      count += g.cta_intersect_compressed(u, adj_u, deg, adj_v);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

