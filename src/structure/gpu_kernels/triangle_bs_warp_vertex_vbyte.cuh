// vertex parallel: each warp takes one vertex
__global__  void //__launch_bounds__(BLOCK_SIZE, 8)
triangle_bs_warp_vertex_vbyte(vidType begin, vidType end, GraphGPU g, vidType *buffer, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
  if (thread_id == 0) printf("running triangle_bs_warp_vertex_vbyte kernel\n");
  AccType count = 0;
  vidType max_deg = g.get_max_degree();
  vidType *adj_v = buffer + (max_deg)*(2*warp_id);
  vidType *adj_u = buffer + (max_deg)*(2*warp_id+1);
  for (vidType v = warp_id + begin; v < end; v += num_warps) {
    auto deg_v = g.decompress_vbyte_warp(v, adj_v);
    for (vidType i = 0; i < deg_v; i++) {
      auto u = adj_v[i];
      auto deg_u = g.decompress_vbyte_warp(u, adj_u);
      count += intersect_num(adj_v, deg_v, adj_u, deg_u);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}


