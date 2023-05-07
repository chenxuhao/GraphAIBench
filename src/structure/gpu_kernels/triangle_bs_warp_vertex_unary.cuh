// vertex parallel: each warp takes one vertex
__global__  void //__launch_bounds__(BLOCK_SIZE, 8)
triangle_bs_warp_vertex_unary(vidType begin, vidType end, GraphTy g, vidType *buffer, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
  AccType count = 0;
  vidType max_deg = g.get_max_degree();
  vidType *adj_v = buffer + (max_deg)*(2*warp_id);
  vidType *adj_u = buffer + (max_deg)*(2*warp_id+1);
  for (vidType v = warp_id + begin; v < end; v += num_warps) {
    vidType deg_v = g.read_degree(v);
    g.decode_unary_warp(v, adj_v, deg_v);
    for (vidType i = 0; i < deg_v; i++) {
      auto u = adj_v[i];
      vidType deg_u = g.read_degree(u);
      g.decode_unary_warp(u, adj_u, deg_u);
      count += intersect_num(adj_v, deg_v, adj_u, deg_u);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

