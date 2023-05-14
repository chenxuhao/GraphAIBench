#include "timer.h"
#include "cutils.h"
#include "aggregator.h"
#include "graph_operations.h"
#include "gpu_device_functions.cuh"

void GAT_Aggregator::init(int l, int nv, int ne, float lr, float drop_rate)
{
  length = l;
  attn_drop = drop_rate;
  assert(attn_drop >= 0. && attn_drop < 1.);
  attn_scale = 1. / (1. - attn_drop);
  auto init_range = sqrt(6.0 / l);
  float_malloc_device(l, d_alpha_l);
  float_malloc_device(l, d_alpha_r);
  rng_uniform_gpu(l, -init_range, init_range, d_alpha_l);
  rng_uniform_gpu(l, -init_range, init_range, d_alpha_r);
  float_malloc_device(l, d_alpha_lgrad);
  float_malloc_device(l, d_alpha_rgrad);
  init_const_gpu(l, 0.0, d_alpha_lgrad);
  init_const_gpu(l, 0.0, d_alpha_rgrad);

  float_malloc_device(ne, d_scores);
  float_malloc_device(ne, d_temp_scores);
  float_malloc_device(ne, d_norm_scores);
  float_malloc_device(ne, d_norm_scores_grad);
  // float_malloc_device(ne, d_trans_norm_scores);
  uint8_malloc_device(ne, d_attn_masks);
  float_malloc_device(ne, d_rands);
  float_malloc_device(nv * l, temp); // avoid repetitive allocation
  epsilon = 0.2;                     // LeakyReLU angle of negative slope

  alpha_opt = new adam(lr);
}

void GAT_Aggregator::aggregate(int len, Graph &g, const float *in, float *out)
{
  Timer t;
  auto nv = g.size();
  auto ne = g.sizeEdges();
  t.Start();
  // std::cout << "GAT aggregator forward: [" << n << " x " << len << "]\n";
  gpu_rng_uniform(ne, d_rands);
  // compute_scores<float>(nv, len, g, epsilon, attn_scale, attn_drop,
  compute_attn_score_warp<float><<<(nv - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(nv, len, g, epsilon, attn_scale, attn_drop,
                                                                                 d_alpha_l, d_alpha_r, d_scores, d_temp_scores, d_norm_scores, d_rands, d_attn_masks, in);
  CudaTest("solving compute_attn_score kernel failed");
  t.Stop();
  time_ops[OP_SCORE] += t.Seconds();

  t.Start();
#ifdef USE_CUSPARSE
  spmm(nv, len, nv, ne, d_norm_scores, (int *)g.row_start_ptr(), (int *)g.edge_dst_ptr(), in, out, temp);
#else
  init_const_gpu(nv * len, 0., out);
  reduce_vertex<float>(nv, len, g, d_norm_scores, in, out);
#endif
  t.Stop();
  time_ops[OP_SPARSEMM] += t.Seconds();
}

void GAT_Aggregator::d_aggregate(int len, Graph &g, const float *feat_in, const float *grad_in, float *grad_out)
{
  Timer t;
  auto nv = g.size();
  auto ne = g.sizeEdges();
  // std::cout << "GAT aggregator backward: [" << nv << " x " << len << "]\n";

  t.Start();
  // compute_scores_grad<float>(nv, len, g, feat_in, grad_in, d_norm_scores_grad);
  compute_scores_grad_warp<float><<<(nv - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(nv, len, g, feat_in, grad_in, d_norm_scores_grad);
  CudaTest("solving compute_scores_grad kernel failed");
  t.Stop();
  time_ops[OP_SCORE] += t.Seconds();

  t.Start();
  init_const_gpu(len, 0.0, d_alpha_lgrad);
  init_const_gpu(len, 0.0, d_alpha_rgrad);
  compute_alpha_grad_warp<float><<<(nv - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(nv, len, g, epsilon, attn_scale, attn_drop,
                                                                                 d_temp_scores, d_norm_scores, d_attn_masks, feat_in, d_norm_scores_grad, d_scores, d_alpha_lgrad, d_alpha_rgrad);
  CudaTest("solving compute_alpha_grad kernel failed");
  t.Stop();
  time_ops[OP_ATTN] += t.Seconds();

  t.Start();
  index_t *rowptr_t, *colidx_t;
  uint_malloc_device(nv + 1, rowptr_t);
  uint_malloc_device(ne, colidx_t);
  csr2csc(nv, nv, ne, d_norm_scores, (const int *)g.row_start_ptr(), (const int *)g.edge_dst_ptr(), d_scores, (int *)rowptr_t, (int *)colidx_t);
  // csr2csc(nv, nv, ne, d_norm_scores, (const int*)g.row_start_ptr(), (const int*)g.edge_dst_ptr(), d_scores, (int*)d_temp_scores, (int*)d_norm_scores);
  // csr2csc(nv, nv, ne, d_norm_scores, (const int*)g.row_start_ptr(), (const int*)g.edge_dst_ptr(), d_trans_norm_scores, (int*)rowptr_t, (int*)colidx_t);
  t.Stop();
  time_ops[OP_TRANSPOSE] += t.Seconds();

  t.Start();
#ifdef USE_CUSPARSE
  spmm(nv, len, nv, ne, d_scores, (int *)g.row_start_ptr(), (int *)g.edge_dst_ptr(), grad_in, grad_out, temp);
  // spmm(nv, len, nv, ne, d_trans_norm_scores, (const int*)g.row_start_ptr(), (const int*)g.edge_dst_ptr(), grad_in, grad_out, temp);
#else
  init_const_gpu(nv * len, 0., grad_out);
  reduce_vertex<float>(nv, len, g, d_scores, grad_in, grad_out);
  // reduce_vertex<float>(nv, len, g, d_trans_norm_scores, grad_in, grad_out);
#endif

  uint_free_device(rowptr_t);
  uint_free_device(colidx_t);
  t.Stop();
  time_ops[OP_SPARSEMM] += t.Seconds();
}

void GAT_Aggregator::update_weights(optimizer *opt)
{
  alpha_opt->update_gpu(length, d_alpha_lgrad, d_alpha_l);
  alpha_opt->update_gpu(length, d_alpha_rgrad, d_alpha_r);
}
