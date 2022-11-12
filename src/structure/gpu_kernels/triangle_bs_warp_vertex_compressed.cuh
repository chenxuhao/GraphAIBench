// vertex parallel: each warp takes one vertex
__global__  void //__launch_bounds__(BLOCK_SIZE, 8)
bs_warp_vertex_compressed(GraphGPU g, vidType *buffer, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
  if (thread_id == 0) printf("running bs_warp_vertex_compressed kernel\n");
  AccType count = 0;
  vidType max_deg = g.get_max_degree();
  vidType *adj_v = buffer + (max_deg)*(2*warp_id);
  vidType *adj_u = buffer + (max_deg)*(2*warp_id+1);
  vidType deg_v = 0, num_itv_v = 0, num_res_v = 0, degree_itv_v = 0;
  for (vidType v = warp_id; v < g.V(); v += num_warps) {
    deg_v = g.warp_decompress(v, adj_v, num_itv_v, num_res_v);
    count += g.intersect_num_warp_compressed_hybrid(adj_v, deg_v, num_itv_v, num_res_v, adj_u);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

// vertex parallel: each warp takes one vertex
__global__  void __launch_bounds__(BLOCK_SIZE, 8)
bs_warp_vertex_compressed_cached(GraphGPU g, vidType *buffer, AccType *total, vidType num_cached) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
  AccType count = 0;
  vidType max_deg = g.get_max_degree();
  vidType *buf1 = buffer + max_deg*(2*warp_id);
  vidType *buf2 = buffer + max_deg*(2*warp_id+1);
  vidType *adj_v, deg_v = 0;
  for (vidType v = warp_id; v < g.V(); v += num_warps) {
    if (v < num_cached) {
      adj_v = g.N(v);
      deg_v = g.get_degree(v);
    } else {
      deg_v = g.warp_decompress(v, buf1);
      adj_v = buf1;
    }
    for (vidType i = 0; i < deg_v; i++) {
      auto u = adj_v[i];
      vidType *adj_u, deg_u = 0;
      if (u < num_cached) {
        adj_u = g.N(u);
        deg_u = g.get_degree(u);
      } else {
        deg_u = g.warp_decompress(u, buf2);
        adj_u = buf2;
      }
      count += intersect_num(adj_v, deg_v, adj_u, deg_u);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

