// edge parallel: each warp takes one edge
__global__ void //__launch_bounds__(BLOCK_SIZE, 8)
hindex_warp_edge(eidType ne, GraphGPU g, vidType *bins, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  __shared__ vidType bin_counts[WARPS_PER_BLOCK*NUM_BUCKETS];

  AccType count = 0;
  for (eidType eid = warp_id; eid < ne; eid += num_warps) {
    auto v = g.get_src(eid);
    auto u = g.get_dst(eid);
    vidType v_size = g.getOutDegree(v);
    vidType u_size = g.getOutDegree(u);
    if (v_size > ADJ_SIZE_THREASHOLD && u_size > ADJ_SIZE_THREASHOLD) {
      count += intersect_warp_hindex(g.N(v), v_size, g.N(u), u_size, bins, bin_counts);
    } else count += intersect_num_bs_cache(g.N(v), v_size, g.N(u), u_size);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

