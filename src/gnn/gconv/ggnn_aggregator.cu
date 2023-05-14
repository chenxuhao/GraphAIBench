#include "timer.h"
#include "aggregator.h"
#include "math_functions.h"
#include "graph_operations.h"

void GGNN_Aggregator::init(int l, int nv, float lr, float drop_rate)
{
  length = l;
  n = nv;
  auto init_range = sqrt(6.0 / l);
  malloc_device<float>(nv * l, d_update_gate);
  malloc_device<float>(nv * nv, d_update_gate_W);
  malloc_device<float>(nv * nv, d_update_gate_U);
  // malloc_device<float>(nv * l, d_update_gate_left);
  // malloc_device<float>(nv * l, d_update_gate_right);
  rng_uniform_gpu(nv * nv, -init_range, init_range, d_update_gate_W);
  rng_uniform_gpu(nv * nv, -init_range, init_range, d_update_gate_U);
  malloc_device<float>(nv * nv, d_update_gate_Wgrad);
  malloc_device<float>(nv * nv, d_update_gate_Ugrad);
  init_const_gpu(nv * nv, 0.0, d_update_gate_Wgrad);
  init_const_gpu(nv * nv, 0.0, d_update_gate_Ugrad);

  malloc_device<float>(nv * l, d_reset_gate);
  malloc_device<float>(nv * nv, d_reset_gate_W);
  malloc_device<float>(nv * nv, d_reset_gate_U);
  // malloc_device<float>(nv * l, d_reset_gate_left);
  // malloc_device<float>(nv * l, d_reset_gate_right);
  // malloc_device<float>(nv * l, d_reset_gate_right_2);
  rng_uniform_gpu(nv * nv, -init_range, init_range, d_reset_gate_W);
  rng_uniform_gpu(nv * nv, -init_range, init_range, d_reset_gate_U);
  malloc_device<float>(nv * nv, d_reset_gate_Wgrad);
  malloc_device<float>(nv * nv, d_reset_gate_Ugrad);
  init_const_gpu(nv * nv, 0.0, d_reset_gate_Wgrad);
  init_const_gpu(nv * nv, 0.0, d_reset_gate_Ugrad);

  malloc_device<float>(nv * l, d_candidate);
  malloc_device<float>(nv * nv, d_candidate_W);
  malloc_device<float>(nv * nv, d_candidate_U);
  // malloc_device<float>(nv * l, d_candidate_left);
  // malloc_device<float>(nv * l, d_candidate_right);
  rng_uniform_gpu(nv * nv, -init_range, init_range, d_candidate_W);
  rng_uniform_gpu(nv * nv, -init_range, init_range, d_candidate_U);
  malloc_device<float>(nv * nv, d_candidate_Wgrad);
  malloc_device<float>(nv * nv, d_candidate_Ugrad);
  init_const_gpu(nv * nv, 0.0, d_candidate_Wgrad);
  init_const_gpu(nv * nv, 0.0, d_candidate_Ugrad);

  malloc_device<float>(n * l, dL_dCv);
  malloc_device<float>(n * l, dL_dHv_temp);
  malloc_device<float>(n * l, dL_dUc_left);
  malloc_device<float>(n * l, dL_dUc_right);
  malloc_device<float>(n * l, dL_dRv);
  malloc_device<float>(n * l, dL_dRv_right);
  malloc_device<float>(n * l, dL_dRv_left);
  malloc_device<float>(n * l, dL_dZv);
  malloc_device<float>(n * l, dL_dZv_left);

  malloc_device<float>(nv * l, d_agg);
  init_const_gpu(nv * l, 0., d_agg);

  opt_reset_U = new adam(lr);
  opt_reset_W = new adam(lr);

  opt_update_U = new adam(lr);
  opt_update_W = new adam(lr);

  opt_candidate_U = new adam(lr);
  opt_candidate_W = new adam(lr);
}

void GGNN_Aggregator::aggregate(int len, Graph &g, float *in, float *out)
{
  Timer t;
  t.Start();
  unsigned n = g.size();
  auto nnz = g.sizeEdges();
  u_int64_t total_blocks = (n - 1) / WARPS_PER_BLOCK + 1;
  dim3 blockDim(32, 32);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

  init_const_gpu(n * len, 0., out);

  // compute a_v
  ggnn_aggregate_all<float><<<total_blocks, BLOCK_SIZE>>>(n, len, g, in, d_agg);
  
  // compute gates and candidate
  ggnn_compute_gates_and_candidate<float><<<gridDim, blockDim>>>(d_update_gate_W, d_reset_gate_W, d_candidate_W, d_update_gate_U, d_reset_gate_U, d_candidate_U, d_agg, in, d_update_gate, d_reset_gate, d_candidate, n, n, len);

  // compute final features
  ggnn_out_warp<float><<<(n - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(n, len, in, d_update_gate, d_candidate, out);

  CudaTest("solving ggnn_out_warp kernel failed");
  t.Stop();
  time_ops[OP_SPARSEMM] += t.Seconds();
}

void GGNN_Aggregator::d_aggregate(int len, Graph &g, float *feat_in, const float *grad_in, float *grad_out)
{
  Timer t;
  t.Start();
  unsigned n = g.size();
  auto nnz = g.sizeEdges();
  u_int64_t total_blocks = (n - 1) / WARPS_PER_BLOCK + 1;
  dim3 blockDim(32, 32);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

  init_const_gpu(n * n, 0.0, d_reset_gate_Ugrad);
  init_const_gpu(n * n, 0.0, d_reset_gate_Wgrad);

  init_const_gpu(n * n, 0.0, d_update_gate_Ugrad);
  init_const_gpu(n * n, 0.0, d_update_gate_Wgrad);

  init_const_gpu(n * n, 0.0, d_candidate_Ugrad);
  init_const_gpu(n * n, 0.0, d_candidate_Wgrad);

  // -------- variables for backpropagation
  // ∂L/∂h_v - grad_in
  // ∂L/∂h_v^{t-1} - grad_out
  // ∂L/∂c_v - candidate gradients
  // ∂L/∂z_v - update gate gradients
  // ∂L/∂r_v - reset gate gradients
  // ----------------------------

  // 1 - ∂L/∂C_v = grad_in ⊙ z_v
  // n X len ⊙ n X len = n X len
  elementwise_kernel<float><<<gridDim, blockDim>>>(grad_in, d_update_gate, dL_dCv, n, len);

  // 2 - ∂L/∂h_v^{t-1} = grad_in ⊙ (1 - z_v) + (grad_in ⊙ z_v) ⊙ ((1 - c_v^2) ⊙ (U_c^T . r_v))
  matmul_kernel<float><<<gridDim, blockDim>>>(d_candidate_U, d_reset_gate, dL_dHv_temp, n, n, len, true, false);
  // matmul(n, len, n, d_candidate_U, d_reset_gate, dL_dHv_temp, true, false);
  ggnn_backward_compute_dL_dHv<float><<<gridDim, blockDim>>>(grad_in, d_update_gate, d_candidate, dL_dHv_temp, grad_out, n, len);

  // 3 - ∂L/∂U_c = (∂L/∂C_v ⊙ (1 - c_v^2)) . (r_v ⊙ (h_v^{t-1}))^T
  ggnn_backward_compute_dL_Uc_left<float><<<gridDim, blockDim>>>(dL_dCv, d_candidate, dL_dUc_left, n, len);
  elementwise_kernel<float><<<gridDim, blockDim>>>(d_reset_gate, feat_in, dL_dUc_right, n, len);
  matmul_kernel<float><<<gridDim, blockDim>>>(dL_dUc_left, dL_dUc_right, d_candidate_Ugrad, n, len, n, false, true);
  // matmul(n, n, len, dL_dUc_left, dL_dUc_right, d_candidate_Ugrad, false, true);

  // 4 - ∂L/∂W_c = (∂L/∂C_v ⊙ (1 - c_v^2)) . (a_v)^T
  matmul_kernel<float><<<gridDim, blockDim>>>(dL_dUc_left, d_agg, d_candidate_Wgrad, n, len, n, false, true);
  // matmul(n, n, len, dL_dUc_left, d_agg, d_candidate_Wgrad, false, true);

  // 5 - ∂L/∂r_v = ∂L/∂C_v ⊙ ((1 - c_v^2) ⊙ (U_c^T . h_v^{t-1})
  matmul_kernel<float><<<gridDim, blockDim>>>(d_candidate_U, feat_in, dL_dRv_right, n, n, len, true, false);
  // matmul(n, len, n, d_candidate_U, feat_in, dL_dRv_right, true, false);
  ggnn_backward_compute_dL_Rv<float><<<gridDim, blockDim>>>(dL_dCv, d_candidate, dL_dRv_right, dL_dRv, n, len);

  // 6 - ∂L/∂U_r = (∂L/∂r_v ⊙ r_v ⊙ (1 - r_v))) . a_v^T
  ggnn_backward_compute_dL_Rv_left<float><<<gridDim, blockDim>>>(dL_dRv, d_reset_gate, dL_dRv_left, n, len);
  matmul_kernel<float><<<gridDim, blockDim>>>(dL_dRv_left, d_agg, d_reset_gate_Ugrad, n, len, n, false, true);
  // matmul(n, n, len, dL_dRv_left, d_agg, d_reset_gate_Ugrad, false, true);

  // 7 - ∂L/∂W_r = (∂L/∂r_v ⊙ r_v ⊙ (1 - r_v))) . (h_v^{t-1})^T
  matmul_kernel<float><<<gridDim, blockDim>>>(dL_dRv_left, feat_in, d_reset_gate_Wgrad, n, len, n, false, true);
  // matmul(n, n, len, dL_dRv_left, feat_in, d_reset_gate_Wgrad, false, true);

  // 8 - ∂L/∂z_v = ∂L/∂h_v ⊙ (-h_v^{t-1} + c_v)
  ggnn_backward_compute_dL_Zv<float><<<gridDim, blockDim>>>(grad_in, feat_in, d_candidate, dL_dZv, n, len);

  // 9 - ∂L/∂U_z = (∂L/∂z_v ⊙ z_v ⊙ (1 - z_v)) . (h_v^{t-1})^T
  ggnn_backward_compute_dL_Zv_left<float><<<gridDim, blockDim>>>(dL_dZv, d_update_gate, dL_dZv_left, n, len);
  matmul_kernel<float><<<gridDim, blockDim>>>(dL_dZv_left, feat_in, d_update_gate_Ugrad, n, len, n, false, true);
  // matmul(n, n, len, dL_dZv_left, feat_in, d_update_gate_Ugrad, false, true);

  // 10 - ∂L/∂W_z = (∂L/∂z_v ⊙ z_v ⊙ (1 - z_v)) . (a_v)^T
  matmul_kernel<float><<<gridDim, blockDim>>>(dL_dZv_left, d_agg, d_update_gate_Wgrad, n, len, n, false, true);
  // matmul(n, n, len, dL_dZv_left, d_agg, d_update_gate_Wgrad, false, true);
}

void GGNN_Aggregator::update_weights(optimizer *opt)
{
  opt_update_W->update_gpu(n * n, d_update_gate_Wgrad, d_update_gate_W);
  opt_update_U->update_gpu(n * n, d_update_gate_Ugrad, d_update_gate_U);

  opt_reset_W->update_gpu(n * n, d_reset_gate_Wgrad, d_reset_gate_W);
  opt_reset_U->update_gpu(n * n, d_reset_gate_Ugrad, d_reset_gate_U);

  opt_candidate_W->update_gpu(n * n, d_candidate_Wgrad, d_candidate_W);
  opt_candidate_U->update_gpu(n * n, d_candidate_Ugrad, d_candidate_U);
}
