__global__ void hindex_warp_edge_compressed(GraphGPU g, vidType *bins, vidType *buffer, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
  __shared__ vidType bin_counts[WARPS_PER_BLOCK*NUM_BUCKETS];
  vidType max_deg = g.get_max_degree();
  vidType *buf1 = buffer + max_deg*(3*warp_id);
  vidType *buf2 = buffer + max_deg*(3*warp_id+1);
  vidType *buf3 = buffer + max_deg*(3*warp_id+2);
  AccType count = 0;
  for (eidType eid = warp_id; eid < g.E(); eid += num_warps) {
    auto v = g.get_src(eid);
    auto u = g.get_dst(eid);
    vidType deg_v = 0, deg_u = 0;
    auto adj_v = g.warp_decompress(v, buf1, NULL, deg_v);
    auto adj_u = g.warp_decompress(u, buf2, NULL, deg_u);
    count += intersect_warp_hindex(adj_v, deg_v, adj_u, deg_u, bins, bin_counts);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

