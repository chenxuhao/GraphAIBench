#include "graph_conv_layer.h"

// Assume before calling forward, feat_in has already been filled in. 
// feat_out should be the feat_in of next layer
void GCN_layer::forward(float* feat_out) {
  auto x = num_samples;
  auto y = dim_in;
  auto z = dim_out;
  //std::cout << "GCN Layer " << level_ << " forward: [" << x << " x " << y << "] to [" << x << " x " << z << "]\n";
#ifdef CSR_SEGMENTING
  //if (y < z) graph->update_feat_len(y);
  //else graph->update_feat_len(z);
#endif
  float* in_data = feat_in;
  if (feat_dropout_rate > 0. && phase_ == net_phase::TRAIN) {
    dropout_cpu(x, y, feat_scale, feat_dropout_rate, in_data, dropout_mask, &in_temp[0]);
    in_data = &in_temp[0]; 
  }
  if (y > z) {
    matmul(x, z, y, in_data, &W_neigh[0], &out_temp[0]); // x*y; y*z; x*z
    aggr.aggregate(z, *graph, &out_temp[0], feat_out); // x*x; x*z; x*z
  } else {
    aggr.aggregate(y, *graph, in_data, &in_temp1[0]); // x*x; x*y; x*y
    matmul(x, z, y, &in_temp1[0], &W_neigh[0], feat_out); // x*y; y*z; x*z
  }
  if (is_bias) bias_mv(x, z, feat_out, &bias[0]);
  if (is_act) relu_cpu(x*z, feat_out, feat_out);
}

// Assume before calling backward, grad_in has already been filled in. 
// grad_out should be the grad_in of previous layer, grad_out = d L / d X^(l-1)
void GCN_layer::backward(float* feat_out, float* grad_out) {
  auto x = num_samples;
  auto y = dim_in;
  auto z = dim_out;
  //std::cout << "GCN Layer " << level_ << " backward: [" << x << " x " << y << "] to [" << x << " x " << z << "]\n";
#ifdef CSR_SEGMENTING
  //if (y < z) graph->update_feat_len(y);
  //else graph->update_feat_len(z);
#endif
  if (is_act) d_relu_cpu(x*z, grad_in, feat_out, grad_in);
  if (is_bias) reduce_sum(x, z, grad_in, bias_grad);
  if (y > z) {
    aggr.d_aggregate(z, *graph, NULL, grad_in, &out_temp[0]); // x*x; x*z; x*z
    if (level_ > 0) {
      matmul(x, y, z, &out_temp[0], &W_neigh[0], grad_out, false, true); // x*z; z*y -> x*y
    }
    float* in_data = feat_in;
    if (feat_dropout_rate > 0.) in_data = &in_temp[0];
    matmul(y, z, x, in_data, &out_temp[0], &W_neigh_grad[0], true, false); // y*x; x*z; y*z
  } else {
    if (level_ > 0) {
      matmul(x, y, z, grad_in, &W_neigh[0], &in_temp[0], false, true); // x*z; z*y; x*y
      aggr.d_aggregate(y, *graph, NULL, &in_temp[0], grad_out); // x*x; x*y; x*y
    }
    matmul(y, z, x, &in_temp1[0], grad_in, &W_neigh_grad[0], true, false); // y*x; x*z; y*z
  }
  if (level_ != 0 && feat_dropout_rate > 0.)
    d_dropout_cpu(x, y, feat_scale, grad_out, dropout_mask, grad_out);
}

void GCN_layer::update_weight(optimizer* opt) {
  opt->update(W_neigh_grad, W_neigh);
  if (use_concat) opt->update(W_self_grad, W_self);
  if (is_bias) opt->update(bias_grad, bias);
}

