#include "aggregator.h"

void GCN_Aggregator::init(int l, int nv, int nb, float, float) {
  length = l;
  //std::cout << "vector length set to: " << length << "\n";
  if (nv > 0 && nb > 0) {
    std::cout << "Allocating partial sums in each aggregator\n";
    partial_sums.resize(nb);
    for (int bid = 0; bid < nb; bid ++) {
      partial_sums[bid].resize(nv);
      for (int vid = 0; vid < nv; vid ++) {
        partial_sums[bid][vid].resize(l);
      }
    }
  }
}

vec_t& GCN_Aggregator::get_partial_feats(int bid, index_t vid) {
  return partial_sums[bid][vid];
}

// aggregation based on graph topology
void GCN_Aggregator::aggregate(int len, Graph& g, const float* in, float* out) {
#ifdef CSR_SEGMENTING
  update_all_blocked(len, g, in, out);
#else
#ifdef PRECOMPUTE_SCORES
  spmm(g.size(), len, g.size(), g.sizeEdges(), g.edge_data_ptr(), (int*)g.row_start_ptr(), (int*)g.edge_dst_ptr(), in, out);
#else
  update_all(len, g, in, out);
#endif
#endif
}

// since graph is symmetric, the derivative is the same
void GCN_Aggregator::d_aggregate(int len, Graph& g, const float*, const float* in, float* out) {
#ifdef CSR_SEGMENTING
  update_all_blocked(len, g, in, out);
#else
#ifdef PRECOMPUTE_SCORES
  spmm(g.size(), len, g.size(), g.sizeEdges(), g.edge_data_ptr(), (int*)g.row_start_ptr(), (int*)g.edge_dst_ptr(), in, out);
#else
  update_all(len, g, in, out); // x*x; x*z -> x*z
#endif
#endif
}

void GCN_Aggregator::update_all(int len, Graph& g, const float* in, float* out) {
  double t1 = omp_get_wtime();
  size_t n = g.size();
  //std::cout << "[update_all] graph size: " << n << ", len = " << len << "\n";
  #pragma omp parallel
  {
  vec_t neighbor(len);
  #pragma omp for schedule(dynamic, 64)
  for (size_t src = 0; src < n; src++) {
    auto src_idx = src * len;
    // zero out the output data
    clear_cpu(len, &out[src_idx]);
    // get normalization factor/score
    auto a = g.get_vertex_data(src);
    //float a = 1.0 / std::sqrt(float(g.get_degree(src)));
    // gather neighbors' embeddings
    for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
      const auto dst = g.getEdgeDst(e);
      float b = a * g.get_vertex_data(dst);
      //float b = a / std::sqrt(float(g.get_degree(dst)));
      // scale the neighbor's data using the normalization factor
      scale(len, b, &in[dst*len], &neighbor[0]);
      // use scaled data to update; out[src] += in[dst]
      vadd_cpu(len, &out[src_idx], &neighbor[0], &out[src_idx]);
    }
  }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_SPARSEMM] += t2 - t1;
}

void GCN_Aggregator::update_all_blocked(int len, Graph& g, const float* in, float* out) {
  double t1 = omp_get_wtime();
  std::fill(out, out+len*g.size(), 0);
  // parallel subgraph processing
  auto num_subgraphs = g.get_num_subgraphs();
  auto num_ranges = g.get_num_ranges();
  //std::cout << "update_all_blocked, len = " << len << ", nb = " << num_subgraphs << ", nr = " << num_ranges << "\n";
  for (int bid = 0; bid < num_subgraphs; bid ++) {
    //std::cout << "BLOCK: " << bid << "\n";
    auto size = g.get_subgraph_size(bid);
    //printf("processing subgraph[%d] with %d vertices\n", bid, size);
    #pragma omp parallel for schedule(dynamic, 64)
    for (index_t u = 0; u < size; u ++) {
      //std::cout << "\t src = " << u << "\n";
      //auto tid = omp_get_thread_num();
      auto& sum = get_partial_feats(bid, u);
      //if (sum.size() != len)
      //  std::cout << "sum vector length " << sum.size() << " not equal to " << len << "\n";
      assert(sum.size() == size_t(len));
      clear_cpu(len, &sum[0]);
      auto row_begin = g.edge_begin_blocked(bid, u);
      auto row_end = g.edge_end_blocked(bid, u);
      auto gid_u = g.get_global_vid(bid, u);
      auto a = g.get_vertex_data(gid_u);
      for (index_t offset = row_begin; offset < row_end; offset ++) {
        index_t v = g.edge_dst_blocked(bid, offset);
        assert(v < g.size());
        //std::cout << "\t\t dst = " << v << "\n";
        //auto gid_v = g.get_global_vid(bid, v);
        float b = a * g.get_vertex_data(v);
        vec_t neighbor(len);
        scale(len, b, &in[v*len], &neighbor[0]);
        vadd_cpu(len, &sum[0], &neighbor[0], &sum[0]);
      }
      //copy_cpu(len, &sums[tid][0], &partial_sum[0]);
      //for (int i = 0; i < len; i ++)
      //  std::cout << "\t sum[" << bid << "][" << u << "][" << i << "] = " << sum[i] << "\n";
    }
  }
  //std::cout << "DEBUG1:\n";
  //for (int i = 0; i < len; i ++)
  //  std::cout << "sum[0][1][" << i << "] = " << get_partial_feats(0, 1)[i] << "\n";

  // cache-aware merge
  #pragma omp parallel for schedule(dynamic, 64)
  for (int rid = 0; rid < num_ranges; rid ++) {
    for (int bid = 0; bid < num_subgraphs; bid ++) {
      auto start = g.get_range_index(bid, rid);
      auto end = g.get_range_index(bid, rid+1);
      for (auto lid = start; lid < end; lid ++) {
        auto gid = g.get_global_vid(bid, lid);
        //std::cout << "lid = " << lid << ", gid = " << gid << "\n";
        auto& local_feats = get_partial_feats(bid, lid);
        vadd_cpu(len, &out[gid*len], &local_feats[0], &out[gid*len]);
        //if (gid == 1) {
        //  std::cout << "DEBUG2: lid = " << lid << "\n";
        //  for (int i = 0; i < len; i ++) {
        //    std::cout << "sum[" << bid << "][" << gid << "][" << i << "] = " << local_feats[i] << "\n";
        //    std::cout << "out[" << gid << "][" << i << "] = " << out[gid*len+i] << "\n";
        //  }
        //}
      }
    }
  }
  //for (size_t v = 0; v < g.size(); v ++)
  //  for (int i = 0; i < len; i ++)
  //   std::cout << "out[" << v << "][" << i << "] = " << out[v*len+i] << "\n";
  double t2 = omp_get_wtime();
  time_ops[OP_SPARSEMM] += t2 - t1;
}

