#include "graph_conv_layer.h"

void GAT_layer::forward(float* feat_out) {
  size_t x = num_samples;
  size_t y = dim_in;
  size_t z = dim_out;

  // dropout
  float* in_data = feat_in;
  if (feat_dropout_rate > 0. && phase_ == net_phase::TRAIN) {
    dropout_cpu(x, y, feat_scale, feat_dropout_rate, in_data, dropout_mask, &in_temp[0]);
    in_data = &in_temp[0]; 
  }
  // linear transformation
  matmul(x, z, y, in_data, &W_neigh[0], &out_temp[0]);
  // aggregation
  aggr.aggregate(z, *graph, &out_temp[0], feat_out);
  // bias
  if (is_bias) bias_mv(x, z, feat_out, &bias[0]);
  // ReLU
  if (is_act) relu_cpu(x * z, feat_out, feat_out);
}

void GAT_layer::backward(float* feat_out, float* grad_out) {
  size_t x = num_samples;
  size_t y = dim_in;
  size_t z = dim_out;

  if (is_act) d_relu_cpu(x * z, grad_in, feat_out, grad_in);
  if (is_bias) reduce_sum(x, z, grad_in, bias_grad);
  float* in_data = feat_in;
  if (feat_dropout_rate > 0.) in_data = &in_temp[0];
  float* out_temp_grad = &out_temp[0]; // feat_out become useless and can be used for intermediate data
  // compute gradients for feature vectors
  aggr.d_aggregate(z, *graph, &out_temp[0], grad_in, out_temp_grad);
  if (level_ != 0) {
    matmul(x, y, z, out_temp_grad, &W_neigh[0], grad_out, false, true); // x*z; z*y -> x*y
    if (feat_dropout_rate > 0.) d_dropout_cpu(x, y, feat_scale, grad_out, dropout_mask, grad_out);
  }
  // compute gradients for weight matrix, i.e. the parameters
  matmul(y, z, x, in_data, out_temp_grad, &W_neigh_grad[0], true); // y*x; x*z; y*z
}

void GAT_layer::update_weight(optimizer* opt) {
  opt->update(W_neigh_grad, W_neigh);
  aggr.update_weights(opt);
  if (is_bias) opt->update(bias_grad, bias);
}

