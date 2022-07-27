// CTA-centric edge parallel: each thread block takes one edge
__global__ void cta_triangle_compressed(vidType nv, GRAPH_TYPE *g, OFFSET_TYPE *offsets, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  AccType count = 0;
  //CgrReader cgrr;
  for (vidType v = blockIdx.x; v < nv; v += gridDim.x) {
    //row_begin = offsets[v];
    //cgrr.init(v, g, row_begin);
    //SIZE_TYPE segment_cnt = cgrr.decode_segment_cnt();
    for (vidType i = 0; i < g.get_degree(); i++) {
      auto u = g.N(v, i);
      count += g.cta_intersect_cache(v, u);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

