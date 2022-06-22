#include "util.h"
#include "aggregator.h"
#include "math_functions.h"
#include "graph_operations.h"

void SAGE_Aggregator::init(int l, int nv, int, float lr, float drop_rate) {
  length = l;
  float_malloc_device(nv*l, temp); // avoid repetitive allocation; TODO: need optimize when sampling
}

void SAGE_Aggregator::aggregate(int len, Graph& g, const float* in, float* out) {
  Timer t;
  t.Start();
  unsigned n = g.size();
#ifdef USE_CUSPARSE
  spmm(n, len, n, g.sizeEdges(), g.edge_data_ptr(), (int*)g.row_start_ptr(), (int*)g.edge_dst_ptr(), in, out, temp);
#else
  init_const_gpu(n*len, 0., out);
  //reduce_vertex_sage(n, len, g, in, out, false);
  update_all_sage<float><<<(n-1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(n, len, g, in, out, false);
  CudaTest("solving update_all_sage kernel failed");
#endif
  t.Stop();
  time_ops[OP_SPARSEMM] += t.Seconds();
}

void SAGE_Aggregator::d_aggregate(int len, Graph& g, const float*, const float* in, float* out) {
  Timer t;
  t.Start();
  unsigned n = g.size();
#ifdef USE_CUSPARSE
  spmm(n, len, n, g.sizeEdges(), g.trans_edge_data_ptr(), (int*)g.row_start_ptr(), (int*)g.edge_dst_ptr(), in, out, temp);
#else
  init_const_gpu(n*len, 0., out);
  //reduce_vertex_sage(n, len, g, in, out, true);
  update_all_sage<float><<<(n-1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(n, len, g, in, out, true);
  CudaTest("solving update_all_sage kernel failed");
#endif
  t.Stop();
  time_ops[OP_SPARSEMM] += t.Seconds();
}

