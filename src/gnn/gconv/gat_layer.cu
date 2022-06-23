#include "graph_conv_layer.h"

void GAT_layer::forward(float* feat_out) {
  size_t x = num_samples;
  size_t y = dim_in;
  size_t z = dim_out;
  // dropout
  float* in_data = feat_in;
  if (feat_dropout_rate > 0. && phase_ == net_phase::TRAIN) {
    dropout_gpu(x*y, feat_scale, feat_dropout_rate, in_data, dropout_mask, d_in_temp);
    in_data = d_in_temp; 
  }
  matmul(x, z, y, in_data, d_W_neigh, d_out_temp);
  aggr.aggregate(z, *graph, d_out_temp, feat_out);
  if (is_bias) bias_mv(x, z, feat_out, d_bias);
  if (is_act) relu_gpu(x * z, feat_out, feat_out);
}

void GAT_layer::backward(float* feat_out, float* grad_out) {
  size_t x = num_samples;
  size_t y = dim_in;
  size_t z = dim_out;

  if (is_act) d_relu_gpu(x * z, grad_in, feat_out, grad_in);
  if (is_bias) reduce_sum(x, z, grad_in, d_bias_grad);
  float* in_data = feat_in;
  if (feat_dropout_rate > 0.) in_data = d_in_temp;
  float* out_temp_grad = d_out_temp; // feat_out become useless and can be used for intermediate data
  aggr.d_aggregate(z, *graph, d_out_temp, grad_in, out_temp_grad);
  if (level_ != 0) {
    matmul(x, y, z, out_temp_grad, d_W_neigh, grad_out, false, true); // x*z; z*y -> x*y
    if (feat_dropout_rate > 0.) 
      d_dropout_gpu(x*y, feat_scale, grad_out, dropout_mask, grad_out);
  }
  matmul(y, z, x, in_data, out_temp_grad, d_W_neigh_grad, true); // y*x; x*z; y*z
}

void GAT_layer::update_weight(optimizer* opt) {
  optm->update_gpu(dim_in * dim_out, d_W_neigh_grad, d_W_neigh);
  if (use_concat) optm->update_gpu(dim_in * dim_out, d_W_self_grad, d_W_self);
  if (is_bias) optm->update_gpu(dim_out, d_bias_grad, d_bias);
  aggr.update_weights(opt);
}

