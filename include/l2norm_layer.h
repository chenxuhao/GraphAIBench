#pragma once
#include "global.h"

class l2norm_layer {
 private:
  int num_samples;
  int dim;
  float *feat_in;
  float *grad_in;
 public:
  l2norm_layer(int nv, int len);
  void forward(float* feat_out);
  void backward(float* grad_out);
  float* get_feat_in() { return feat_in; }
  float* get_grad_in() { return grad_in; } 
  void update_dim_size(int sz);
};
