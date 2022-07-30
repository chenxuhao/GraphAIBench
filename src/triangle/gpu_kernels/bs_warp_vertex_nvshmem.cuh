// vertex paralle: each warp takes one vertex
__global__ void warp_vertex_nvshmem(vidType begin, vidType end, GraphGPU g, int mype, int npes, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  AccType count = 0;
  vidType *buffer = g.get_buffer_ptr();
  int subgraph_size = (g.V()-1) / g.get_num_devices() + 1;
 
  for (auto v = warp_id+begin; v < end; v += num_warps) {
    vidType *v_ptr = g.N(v);
    vidType v_size = g.getOutDegree(v);
    for (auto e = 0; e < v_size; e ++) {
      auto u = v_ptr[e];
      vidType u_size = g.getOutDegree(u);
      vidType *u_ptr = NULL;
      if (u < begin || u >= end) {
        auto pe = u / subgraph_size;
        auto u_start = g.edge_begin(u);
        auto first_vertex = pe * subgraph_size;
        auto start = g.edge_begin(first_vertex);
        auto offset = u_start - start;
        nvshmem_int_get(buffer, g.colidx()+offset, u_size, pe);
        u_ptr = buffer;
      } else {
        u_ptr = g.N(u);
      }
      count += intersect_num(v_ptr, v_size, u_ptr, u_size);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if(threadIdx.x == 0) atomicAdd(total, block_num);
}

