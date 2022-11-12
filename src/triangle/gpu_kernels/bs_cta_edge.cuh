// CTA-centric edge parallel: each thread block takes one edge
__global__ void triangle_bs_cta_edge(GraphGPU g, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  AccType count = 0;
  __shared__ vidType v, u;
  for (eidType eid = blockIdx.x; eid < g.E(); eid += gridDim.x) {
    if (threadIdx.x == 0) {
      v = g.get_src(eid);
      u = g.get_dst(eid);
    }
    __syncthreads();
    count += g.cta_intersect(v, u);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

