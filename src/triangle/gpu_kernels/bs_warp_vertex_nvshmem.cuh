// vertex paralle: each warp takes one vertex
__global__ void warp_vertex_nvshmem(vidType begin, vidType end, GraphGPU g, vidType *buffers, int mype, int npes, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1);           // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;               // warp index within the CTA
 
  AccType count = 0;
  vidType *buffer  = &buffers[int64_t(warp_id)*int64_t(max_deg)];
  int subgraph_size = (g.V()-1) / npes + 1;
  auto start_idx = g.edge_begin(begin);
  //if (thread_id == 0) printf("mype=%d, npes=%d, begin=%d, end=%d, subgraph_size=%d\n", mype, npes, begin, end, subgraph_size);
  for (auto v = warp_id+begin; v < end; v += num_warps) {
    auto v_begin = g.edge_begin(v);
    vidType *v_ptr = g.colidx()+(v_begin-start_idx);
    vidType v_size = g.getOutDegree(v);
    for (auto e = 0; e < v_size; e ++) {
      auto u = v_ptr[e];
      vidType u_size = g.getOutDegree(u);
      auto u_begin = g.edge_begin(u);
      auto pe = u / subgraph_size;
      auto u_start_idx = g.edge_begin(pe * subgraph_size);
      auto u_offset = u_begin - u_start_idx;
      vidType *u_ptr = g.colidx() + u_offset;
      //if (thread_lane == 0) printf("v=%d, u=%d, v_deg=%d, u_deg=%d, u_pe=%d, u_offset=%d\n", v, u, v_size, u_size, pe, u_offset);
      if (pe != mype) { // remote data fetch
        //if (thread_lane == 0) printf("remote data access: u=%d, pe=%d, u_deg=%d\n", u, pe, u_size);
        nvshmemx_int_get_warp(buffer, u_ptr, u_size, pe);
        u_ptr = buffer;
      }
      //if (thread_lane == 0) {
      //  printf("u(%d)'s neighbors: [", u);
      //  for (int i = 0; i < u_size; i++) printf("%d ", u_ptr[i]);
      //  printf("]\n");
      //}
      auto n_tri = intersect_num(v_ptr, v_size, u_ptr, u_size);
      count += n_tri;
      //if (thread_lane == 0) printf("v=%d, u=%d, count: %d\n", v, u, count);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

