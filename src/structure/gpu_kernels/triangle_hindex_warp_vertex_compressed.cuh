// vertex parallel: each warp takes one vertex
__global__  //__launch_bounds__(BLOCK_SIZE, 3)
void hindex_warp_vertex_compressed(GraphGPU g, vidType *bins, vidType *buffer, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  __shared__ vidType bin_counts[WARPS_PER_BLOCK*NUM_BUCKETS];

  AccType count = 0;
  vidType max_deg = g.get_max_degree();
  vidType *buf1 = buffer + max_deg*(3*warp_id);
  vidType *buf2 = buffer + max_deg*(3*warp_id+1);
  vidType *buf3 = buffer + max_deg*(3*warp_id+2);
  vidType *adj_v, deg_v = 0;
 
  for (vidType v = warp_id; v < g.V(); v += num_warps) {
    adj_v = g.warp_decompress(v, buf1, NULL, deg_v);
    for (vidType i = 0; i < deg_v; i++) {
      auto u = adj_v[i];
      vidType *adj_u, deg_u = 0;
      //if (adj_v == buf2) adj_u = g.warp_decompress(u, buf1, buf3, deg_u);
      //else
        adj_u = g.warp_decompress(u, buf2, NULL, deg_u);
      count += intersect_warp_hindex(adj_v, deg_v, adj_u, deg_u, bins, bin_counts);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

