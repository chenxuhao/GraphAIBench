#include "util.h"
#include "aggregator.h"
#include "math_functions.h"
#include "graph_operations.h"

void GCN_Aggregator::init(int l, int nv, int ne, float lr, float drop_rate) {
  length = l;
  if (nv > n) {
    if (temp) float_free_device(temp);
    float_malloc_device(nv * l, temp); // avoid repetitive allocation
  }
  n = nv;
}

void GCN_Aggregator::aggregate(int len, Graph& g, const float* in, float* out) {
  Timer t;
  t.Start();
  unsigned n = g.size();
  auto nnz = g.sizeEdges();
#ifdef USE_CUSPARSE
  spmm(n, len, n, nnz, g.edge_data_ptr(), (int*)g.row_start_ptr(), (int*)g.edge_dst_ptr(), in, out, temp);
#else
  init_const_gpu(n*len, 0., out);
  //reduce_vertex<float>(n, len, g, g.edge_data_ptr(), in, out);
  //update_all_gcn<float><<<(n-1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(n, len, g, in, out);
  reduce_warp<float><<<(n-1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(n, len, g, g.edge_data_ptr(), in, out);
  //reduce_cta<float><<<(n-1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(n, len, g, g.edge_data_ptr(), in, out);
  CudaTest("solving update_all_gcn kernel failed");
#endif
  t.Stop();
  time_ops[OP_SPARSEMM] += t.Seconds();
}

// since graph is symmetric, the derivative is the same
void GCN_Aggregator::d_aggregate(int len, Graph& g, const float*, const float* in, float* out) {
  Timer t;
  t.Start();
  unsigned n = g.size();
  auto nnz = g.sizeEdges();
#ifdef USE_CUSPARSE
  spmm(n, len, n, nnz, g.edge_data_ptr(), (int*)g.row_start_ptr(), (int*)g.edge_dst_ptr(), in, out, temp);
#else
  init_const_gpu(n*len, 0., out);
  //reduce_vertex<float>(n, len, g, g.edge_data_ptr(), in, out);
  //update_all_gcn<float><<<(n-1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(n, len, g, in, out);
  reduce_warp<float><<<(n-1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(n, len, g, g.edge_data_ptr(), in, out);
  //reduce_cta<float><<<(n-1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(n, len, g, g.edge_data_ptr(), in, out);
  CudaTest("solving update_all_gcn kernel failed");
#endif
  t.Stop();
  time_ops[OP_SPARSEMM] += t.Seconds();
}


