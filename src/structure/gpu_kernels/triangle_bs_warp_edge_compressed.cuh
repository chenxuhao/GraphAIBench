// warp-wise edge-parallel: each warp takes one edge
__global__ void //__launch_bounds__(BLOCK_SIZE, 8)
bs_warp_edge_compressed(GraphGPU g, vidType *buffer, AccType *total, vidType num_cached) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id   = thread_id   / WARP_SIZE;               // global warp index
  int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  AccType count = 0;
  vidType max_deg = g.get_max_degree();
  vidType *buf1 = buffer + max_deg*(2*warp_id);
  vidType *buf2 = buffer + max_deg*(2*warp_id+1);
 
  for (eidType eid = warp_id; eid < g.E(); eid += num_warps) {
    auto v = g.get_src(eid);
    auto u = g.get_dst(eid);
#if 0
    vidType deg_v = 0, deg_u = 0;
    vidType *adj_v, *adj_u;
    if (v < num_cached) {
      adj_v = g.N(v);
      deg_v = g.get_degree(v);
    } else adj_v = g.warp_decompress(v, buf1, NULL, deg_v);
    if (u < num_cached) {
      adj_u = g.N(u);
      deg_u = g.get_degree(u);
    } else adj_u = g.warp_decompress(u, buf2, NULL, deg_u);
    count += intersect_num(adj_v, deg_v, adj_u, deg_u);
#else
    count += g.intersect_num_warp_compressed(v, u, buf1, buf2);
#endif
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

