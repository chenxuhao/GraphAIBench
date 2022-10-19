
__forceinline__ __device__ void init_bin_counts(int id, int offset, vidType *bin_counts) {
  for (auto i = id + offset; i < offset + NUM_BUCKETS; i += WARP_SIZE)
    bin_counts[i] = 0;
  __syncwarp();
}

// edge parallel: each warp takes one edge
__global__ void hindex_warp_vertex_compressed(GraphGPU g, vidType *bins, vidType *buffer, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int bin_offset   = warp_lane * NUM_BUCKETS;
  __shared__ vidType bin_counts[WARPS_PER_BLOCK*NUM_BUCKETS];

  AccType count = 0;
  vidType *buf1 = buffer + max_deg*(3*warp_id);
  vidType *buf2 = buffer + max_deg*(3*warp_id+1);
  vidType *buf3 = buffer + max_deg*(3*warp_id+2);
  vidType *adj_v, deg_v = 0;
 
  for (vidType v = warp_id; v < g.V(); v += num_warps) {
    adj_v = g.warp_decompress(v, buf1, buf2, deg_v);
    assert(deg_v == g.get_degree(v));
    for (vidType i = 0; i < deg_v; i++) {
      auto u = adj_v[i];
      //if (u > v) continue;
      vidType *adj_u, deg_u = 0;
      if (adj_v == buf2) {
        adj_u = g.warp_decompress(u, buf1, buf3, deg_u);
      } else {
        adj_u = g.warp_decompress(u, buf2, buf3, deg_u);
      }
      assert(deg_u == g.get_degree(u));
      init_bin_counts(thread_lane, bin_offset, bin_counts); // ensure bit counts are empty
      count += intersect_warp_hindex(adj_v, deg_v, adj_u, deg_u, bins, bin_counts);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

// edge parallel: each warp takes one edge
__global__ void hindex_warp_vertex(GraphGPU g, vidType *bins, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id     = thread_id   / WARP_SIZE;                // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  int bin_offset   = warp_lane * NUM_BUCKETS;
  __shared__ vidType bin_counts[WARPS_PER_BLOCK*NUM_BUCKETS];

  AccType count = 0;
  vidType *adj_v, deg_v = 0;
  for (vidType v = warp_id; v < g.V(); v += num_warps) {
    adj_v = g.N(v);
    deg_v = g.get_degree(v);
    for (vidType i = 0; i < deg_v; i++) {
      auto u = adj_v[i];
      if (u > v) break;
      vidType *adj_u = g.N(u);
      vidType deg_u = g.get_degree(u);
      init_bin_counts(thread_lane, bin_offset, bin_counts); // ensure bit counts are empty
      count += intersect_warp_hindex(adj_v, deg_v, adj_u, deg_u, bins, bin_counts, u);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

