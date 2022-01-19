#pragma once
#include "optimizer.h"

class dense_layer {
 private:
  bool is_bias;
  int num_samples;
  int dim_in;
  int dim_out;
  float *feat_in;
  float *grad_in;
  vec_t weight;
  vec_t weight_grad;
  vec_t bias;
  vec_t bias_grad;
  optimizer* optm;
  float* d_weight;
  float* d_weight_grad;
  float* d_bias;
  float* d_bias_grad;
 public:
  dense_layer(int nv, int in_len, int out_len, float lr);
  void forward(float* feat_out);
  void backward(float* grad_out);
  float* get_feat_in() { return feat_in; }
  float* get_grad_in() { return grad_in; } 
  void update_dim_size(int sz);
};
