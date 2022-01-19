#include "dense_layer.h"
#include "math_functions.h"

dense_layer::dense_layer(int nv, int in_len, int out_len,float lr) :
    is_bias(false), num_samples(nv), dim_in(in_len), dim_out(out_len) {
#ifdef ENABLE_GPU
  float_malloc_device(nv*dim_in, feat_in);
  float_malloc_device(nv*dim_out, grad_in);
  init_const_gpu(nv*dim_in, 0.0, feat_in);
  init_const_gpu(nv*dim_out, 0.0, grad_in);
  float_malloc_device(dim_in*dim_out, d_weight);
  auto init_range = sqrt(6.0 / (dim_in + dim_out));
  // Glorot & Bengio (AISTATS 2010)
  rng_uniform_gpu(dim_in*dim_out, -init_range, init_range, d_weight);
  float_malloc_device(dim_in*dim_out, d_weight_grad);
  init_const_gpu(dim_in*dim_out, 0.0, d_weight_grad);
  if (is_bias) {
    float_malloc_device(dim_out, d_bias);
    float_malloc_device(dim_out, d_bias_grad);
    init_const_gpu(dim_out, 0.0, d_bias);
    init_const_gpu(dim_out, 0.0, d_bias_grad);
  }
#else
  feat_in = new float[nv*dim_in];
  grad_in = new float[nv*dim_out];
  std::fill(feat_in, feat_in+nv*dim_in, 0);
  std::fill(grad_in, grad_in+nv*dim_out, 0);
  weight.resize(dim_in*dim_out);
  weight_grad.resize(dim_in*dim_out);
  init_glorot(dim_in, dim_out, weight, 1); // init weight
  std::fill(weight_grad.begin(), weight_grad.end(), 0);
  if (is_bias) {
    bias.resize(dim_out);
    bias_grad.resize(dim_out);
    std::fill(bias.begin(), bias.end(), 0);
    std::fill(bias_grad.begin(), bias_grad.end(), 0);
  }
#endif
  optm = new adam(lr);
}

void dense_layer::forward(float* feat_out) {
  size_t x = num_samples;
  size_t y = dim_in;
  size_t z = dim_out;
#ifdef ENABLE_GPU
  matmul(x, z, y, feat_in, d_weight, feat_out); // x*y; y*z; x*z
  if (is_bias) bias_mv(x, z, feat_out, d_bias);
#else
  matmul(x, z, y, feat_in, &weight[0], feat_out);
  if (is_bias) bias_mv(x, z, feat_out, &bias[0]);
#endif
}

void dense_layer::backward(float* grad_out) {
  size_t x = num_samples;
  size_t y = dim_in;
  size_t z = dim_out;
#ifdef ENABLE_GPU
  matmul(y, z, x, feat_in, grad_in, d_weight_grad, true); // y*x; x*z; y*z
  matmul(x, y, z, grad_in, d_weight, grad_out, false, true); // x*z; z*y -> x*y
  if (is_bias) reduce_sum(x, z, grad_in, d_bias_grad);
  optm->update_gpu(dim_in*dim_out, d_weight_grad, d_weight);
  if (is_bias) optm->update_gpu(dim_out, d_bias_grad, d_bias);
#else
  matmul(y, z, x, feat_in, grad_in, &weight_grad[0], true); // y*x; x*z; y*z
  matmul(x, y, z, grad_in, &weight[0], grad_out, false, true); // x*z; z*y -> x*y
  if (is_bias) reduce_sum(x, z, grad_in, bias_grad);
  optm->update(weight_grad, weight);
  if (is_bias) optm->update(bias_grad, bias);
#endif
} 

void dense_layer::update_dim_size(int x) {
  if (x > num_samples) {
#ifdef ENABLE_GPU
    if (feat_in) float_free_device(feat_in);
    float_malloc_device(x*dim_in, feat_in);
    init_const_gpu(x*dim_in, 0.0, feat_in);
    //if (grad_in) float_free_device(grad_in);
    //float_malloc_device(x*dim_out, grad_in);
    //init_const_gpu(x*dim_out, 0.0, grad_in);
#else
    if (feat_in) delete[] feat_in;
    feat_in = new float[x*dim_in];
    std::fill(feat_in, feat_in+x*dim_in, 0);
    //if (grad_in) delete[] grad_in;
    //grad_in = new float[x*dim_out];
    //std::fill(grad_in, grad_in+x*dim_out, 0);
#endif
  }
  num_samples = x;
}
