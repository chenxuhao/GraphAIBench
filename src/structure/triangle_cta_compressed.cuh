// CTA-centric vertex parallel: each thread block takes one vertex
__global__ void cta_vertex_compressed(GraphGPU g, vidType *buffer, vidType max_deg, AccType *total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  AccType count = 0;
  vidType *buf1 = buffer + max_deg*(3*blockIdx.x);
  vidType *buf2 = buffer + max_deg*(3*blockIdx.x+1);
  vidType *buf3 = buffer + max_deg*(3*blockIdx.x+2);
  vidType *adj_v, deg_v = 0;
  for (vidType v = blockIdx.x; v < g.V(); v += gridDim.x) {
    adj_v = g.cta_decompress(v, buf1, buf2, deg_v);
    //if (threadIdx.x == 0) printf("v %d, v_deg %d\n", v, deg_v);
    assert(deg_v == g.get_degree(v));
    for (vidType i = 0; i < deg_v; i++) {
      auto u = adj_v[i];
      //__syncthreads();
      vidType *adj_u, deg_u = 0;
      if (adj_v == buf2) {
        adj_u = g.cta_decompress(u, buf1, buf3, deg_u);
      } else {
        adj_u = g.cta_decompress(u, buf2, buf3, deg_u);
      }
      /*
      if (deg_u != g.get_degree(u)) printf("ERROR tid %d, u %d, u_deg %d, deg_u %d\n", threadIdx.x, u, g.get_degree(u), deg_u);
      assert(deg_u == g.get_degree(u));
      if (threadIdx.x == 0) printf("\t u %d, u_deg %d\n", u, deg_u);
      if (threadIdx.x == 0) {
        printf("\t adj_u: [ ");
        for (vidType i = 0; i < deg_u; i++) {
          printf("%d ", adj_u[i]);
        }
        printf("] \n");
      }
      */
      count += intersect_num_cta(adj_u, deg_u, adj_v, deg_v);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(total, block_num);
}

