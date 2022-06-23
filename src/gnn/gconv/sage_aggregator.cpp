#include "aggregator.h"

void SAGE_Aggregator::init(int l, int, int, float, float) {
  length = l;
}

void SAGE_Aggregator::aggregate(int len, Graph& g, const float* in, float* out) {
  double t1 = omp_get_wtime();
  size_t n = g.size();
  //std::cout << "[sage_aggregator] graph size: " << n << ", len = " << length << "\n";
  #pragma omp parallel
  {
  vec_t neighbor(len);
  #pragma omp for schedule(dynamic, 64)
  for (size_t src = 0; src < n; src++) {
    auto src_idx = src * len;
    clear_cpu(len, &out[src_idx]);
    float b = 1.0 / float(g.get_degree(src));
    for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
      auto dst = g.getEdgeDst(e);
      auto dst_idx = dst * len;
      scale(len, b, &in[dst_idx], &neighbor[0]);
      vadd_cpu(len, &out[src_idx], &neighbor[0], &out[src_idx]);
      //scaled_vadd_cpu(len, b, &in[dst_idx], &out[src_idx], &out[src_idx]);
    }
  }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_SPARSEMM] += t2 - t1;
}

void SAGE_Aggregator::d_aggregate(int len, Graph& g, const float*, const float* in, float* out) {
  double t1 = omp_get_wtime();
  size_t n = g.size();
  #pragma omp parallel
  {
  vec_t neighbor(len);
  #pragma omp for schedule(dynamic, 64)
  for (size_t src = 0; src < n; src++) {
    auto src_idx = src * len;
    clear_cpu(len, &out[src_idx]);
    for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
      auto dst = g.getEdgeDst(e);
      float b = 1.0 / float(g.get_degree(dst));
      auto dst_idx = dst * len;
      scale(len, b, &in[dst_idx], &neighbor[0]);
      vadd_cpu(len, &out[src_idx], &neighbor[0], &out[src_idx]);
      //scaled_vadd_cpu(len, b, &in[dst_idx], &out[src_idx], &out[src_idx]);
    }
  }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_SPARSEMM] += t2 - t1;
}

