#include "aggregator.h"
#include "math_functions.h"

void GAT_Aggregator::init(int l, int nv, int ne, float lr, float drop_rate) {
  length = l;
  attn_drop = drop_rate;
  assert(attn_drop>=0. && attn_drop<1.);
  attn_scale = 1. / (1. - attn_drop);
  //std::cout << "score_scale: " << attn_scale << "\n";
  alpha_l.resize(l);
  alpha_r.resize(l);
  init_glorot(l, 1, alpha_l, 2);
  init_glorot(l, 1, alpha_r, 3);
  alpha_lgrad.resize(l);
  alpha_rgrad.resize(l);
  scores.resize(ne); // a score for each edge
  temp_scores.resize(ne);
  //scores_grad.resize(ne);
  norm_scores.resize(ne);
  norm_scores_grad.resize(ne);
  //trans_norm_scores.resize(ne);
  attn_masks.resize(ne);
  epsilon = 0.2; // LeakyReLU angle of negative slope
  alpha_opt = new adam(lr);
}

inline void update_all(int len, Graph& g, const float* scores, const float* in, float* out) {
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
      const auto dst = g.getEdgeDst(e);
      scale(len, scores[e], &in[dst*len], &neighbor[0]);
      vadd_cpu(len, &out[src_idx], &neighbor[0], &out[src_idx]);
    }
  }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_SPARSEMM] += t2 - t1;
}

// `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>` 
// NOTE: GAT paper uses "first concatenation then linear projection"
//  to compute attention scores, while ours is "first projection then
//  addition", the two approaches are mathematically equivalent:
//  We decompose the weight vector a mentioned in the paper into
//  [a_l || a_r], then  a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
//  Our implementation is much efficient because we do not need to
//  save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
//  addition could be optimized with DGL's built-in function u_add_v,
//  which further speeds up computation and saves memory footprint.
void GAT_Aggregator::aggregate(int len, Graph& g, const float* in, float* out) {
  double t1 = omp_get_wtime();
  #pragma omp parallel
  {
  #pragma omp for schedule(dynamic, 64)
  for (size_t src = 0; src < g.size(); src++) {
    auto src_idx = src * len;
    auto begin = g.edge_begin(src);
    auto end = g.edge_end(src);
    // concatenation, dot product, LeakyReLU
    // alpha: learnable weight vector (shared by all vertices)
    auto src_score = dot(len, &alpha_l[0], &in[src_idx]);
    for (auto e = begin; e != end; e++) {
      auto dst = g.getEdgeDst(e);
      auto dst_score = dot(len, &alpha_r[0], &in[dst*len]);
      temp_scores[e] = src_score + dst_score;
      leaky_relu(epsilon, temp_scores[e], scores[e]);
    }
    // softmax to normalize the attention scores on each vertex’s incoming edges
    auto deg_src = g.get_degree(src);
    softmax(deg_src, &scores[begin], &norm_scores[begin]);
    //if (attn_drop > 0.) 
    //  dropout(deg_src, attn_scale, attn_drop, &norm_scores[begin], &attn_masks[begin], &norm_scores[begin]);
/*
    // aggregation: scaled by the attention scores
    clear_cpu(len, &out[src_idx]);
    for (auto e = begin; e != end; e++) {
      auto dst = g.getEdgeDst(e);
      auto dst_idx = dst * len;
      auto score = norm_scores[e];
      scale(len, score, &in[dst_idx], &neighbor[0]);
      vadd_cpu(len, &out[src_idx], &neighbor[0], &out[src_idx]);
    }
*/
  }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_SCORE] += t2 - t1;

  update_all(len, g, &norm_scores[0], &in[0], &out[0]);
}

void GAT_Aggregator::d_aggregate(int len, Graph& g, const float* feat_in, const float* grad_in, float* grad_out) {
  size_t n = g.size();
  double t1 = omp_get_wtime();
  // To compute gradients for the learnable vector `alpha`,
  // we have to first compute gradients for normalized scores.
  // For each edge `e(i,j)`, compute dot product of grad_in[i] and feat_in[j]
  // FW: A*X=Y; BW: A'=Y'*(X^T)
  #pragma omp parallel for schedule(dynamic, 64)
  for (size_t src = 0; src < n; src++) {
    auto src_idx = src * len;
    for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
      auto dst = g.getEdgeDst(e);
      norm_scores_grad[e] = dot(len, &grad_in[src_idx], &feat_in[dst*len]);
    }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_SCORE] += t2 - t1;

  // FW: alpha_l * feat_in[src] = src_score
  // BW: alpha_lgrad = feat_in[src] * src_score_grad
  // FW: alpha_r * feat_in[dst] = dst_score
  // BW: alpha_rgrad = feat_in[dst] * dst_score_grad
  t1 = omp_get_wtime();
  std::fill(alpha_lgrad.begin(), alpha_lgrad.end(), 0);
  std::fill(alpha_rgrad.begin(), alpha_rgrad.end(), 0);
  vec_t local_lgrad(56*len), local_rgrad(56*len);
  #pragma omp parallel
  {
  vec_t temp(len);
  vec_t sum_l(len);
  vec_t sum_r(len);
  clear_cpu(len, &sum_l[0]);
  clear_cpu(len, &sum_r[0]);
  #pragma omp for schedule(dynamic, 64)
  for (size_t src = 0; src < n; src++) {
    auto src_idx = src * len;
    auto begin = g.edge_begin(src);
    auto end = g.edge_end(src);
    auto deg_src = g.get_degree(src);
    //if (attn_drop > 0.) 
    //  d_dropout(deg_src, attn_scale, &norm_scores_grad[begin], &attn_masks[begin], &norm_scores_grad[begin]);
    d_softmax((int)deg_src, &norm_scores[begin], &norm_scores_grad[begin], &scores[begin]);
    float src_score_grad = 0;
    for (auto e = begin; e != end; e++) {
      auto dst = g.getEdgeDst(e);
      float temp_score_grad = 0.;
      temp_score_grad = scores[e] * (temp_scores[e] > 0.0 ? 1.0 : epsilon);
      scale(len, temp_score_grad, &feat_in[dst*len], &temp[0]);
      vadd_cpu(len, &temp[0], &sum_r[0], &sum_r[0]);
      src_score_grad += temp_score_grad;
    }
    scale(len, src_score_grad, &feat_in[src_idx], &temp[0]);
    vadd_cpu(len, &temp[0], &sum_l[0], &sum_l[0]);
    //scaled_vadd_cpu(len, src_score_grad, &feat_in[src_idx], &sum_l[0], &sum_l[0]);
  }
  auto tid = omp_get_thread_num();
  for (int i = 0; i < len; ++i) {
    local_lgrad[tid*len+i] += sum_l[i];
    local_rgrad[tid*len+i] += sum_r[i];
  }
  }
  for (int i = 0; i < 56; ++i) {
    for (int j = 0; j < len; ++j) {
      alpha_lgrad[j] += local_lgrad[i*len+j];
      alpha_rgrad[j] += local_rgrad[i*len+j];
    }
  }
  t2 = omp_get_wtime();
  time_ops[OP_ATTN] += t2 - t1;

  // Compute derivative of aggregation: the graph (adjacency matrix) should be transposed;
  // Note that the graph is undirected (structurally symmetric), 
  // but values are not the same for the symmetric positions
  t1 = omp_get_wtime();
  float* scores_ptr;
  //float* scores_ptr = &trans_norm_scores[0];
  symmetric_csr_transpose(n, g.sizeEdges(), (int*)g.row_start_ptr(), (int*)g.edge_dst_ptr(), &norm_scores[0], scores_ptr);
  t2 = omp_get_wtime();
  time_ops[OP_TRANSPOSE] += t2 - t1;

  /*
  #pragma omp parallel
  {
  vec_t neighbor(len);
  #pragma omp for schedule(dynamic, 64)
  for (size_t src = 0; src < n; src++) {
    auto src_idx = src * len;
    auto src_begin = g.edge_begin(src);
    clear_cpu(len, &grad_out[src_idx]);
    for (auto e = src_begin; e != g.edge_end(src); e++) {
      auto dst = g.getEdgeDst(e);
      auto dst_idx = dst * len;
      auto score = trans_norm_scores[e]; // transposed
      scale(len, score, &grad_in[dst_idx], &neighbor[0]);
      vadd_cpu(len, &grad_out[src_idx], &neighbor[0], &grad_out[src_idx]);
    }
  }
  }
  */
  update_all(len, g, scores_ptr, &grad_in[0], &grad_out[0]);
  delete scores_ptr;
}

void GAT_Aggregator::update_weights(optimizer *opt) {
  alpha_opt->update(alpha_lgrad, alpha_l);
  alpha_opt->update(alpha_rgrad, alpha_r);
}

