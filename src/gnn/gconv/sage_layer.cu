#include "math_functions.h"
#include "graph_conv_layer.h"

void SAGE_layer::forward(float* feat_out) {
  auto x = num_samples;
  auto y = dim_in;
  auto z = dim_out;
  float* in_data = feat_in;
  if (feat_dropout_rate > 0. && phase_ == net_phase::TRAIN) {
    dropout_gpu(x*y, feat_scale, feat_dropout_rate, in_data, dropout_mask, d_in_temp);
    in_data = d_in_temp;
  }
  if (y > z) {
    matmul(x, z, y, in_data, d_W_neigh, d_out_temp); // x*y; y*z; x*z
    aggr.aggregate(z, *graph, d_out_temp, feat_out); // x*x; x*z; x*z
  } else {
    aggr.aggregate(y, *graph, in_data, d_in_temp1); // x*x; x*y; x*y
    matmul(x, z, y, d_in_temp1, d_W_neigh, feat_out); // x*y; y*z; x*z
  }
  matmul(x, z, y, in_data, d_W_self, feat_out, false, false, true); // x*y; y*z; x*z
  if (is_bias) bias_mv(x, z, feat_out, d_bias);
  if (is_act) relu_gpu(x*z, feat_out, feat_out);
}

void SAGE_layer::backward(float* feat_out, float* grad_out) {
  auto x = num_samples;
  auto y = dim_in;
  auto z = dim_out;
  if (is_act) d_relu_gpu(x*z, grad_in, feat_out, grad_in);
  if (is_bias) reduce_sum(x, z, grad_in, d_bias_grad);
  float* in_data = feat_in;
  if (feat_dropout_rate > 0.) in_data = d_in_temp;
  matmul(y, z, x, in_data, grad_in, d_W_self_grad, true, false); // y*x; x*z; y*z
  if (y > z) {
    aggr.d_aggregate(z, *graph, NULL, grad_in, d_out_temp); // x*x; x*z; x*z
    if (level_ > 0)
      matmul(x, y, z, d_out_temp, d_W_neigh, grad_out, false, true); // x*z; z*y -> x*y
    matmul(y, z, x, in_data, d_out_temp, d_W_neigh_grad,true, false); // y*x; x*z; y*z
  } else {
    if (level_ > 0) {
      matmul(x, y, z, grad_in, d_W_neigh, d_in_temp, false, true);
      aggr.d_aggregate(y, *graph, NULL, d_in_temp, grad_out);
    }
    matmul(y, z, x, d_in_temp1, grad_in, d_W_neigh_grad, true, false);
  }
  if (level_ > 0) matmul(x, y, z, grad_in, d_W_self, grad_out, false, true, true); // accumulated MxM
  if (level_ != 0 && feat_dropout_rate > 0.)
    d_dropout_gpu(x*y, feat_scale, grad_out, dropout_mask, grad_out);
}

void SAGE_layer::update_weight(optimizer* opt) {
  optm->update_gpu(dim_in * dim_out, d_W_neigh_grad, d_W_neigh);
  if (use_concat) optm->update_gpu(dim_in * dim_out, d_W_self_grad, d_W_self);
  if (is_bias) optm->update_gpu(dim_out, d_bias_grad, d_bias);
}

