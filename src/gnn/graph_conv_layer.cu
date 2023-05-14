#include "math_functions.hh"
#include "graph_conv_layer.h"

template <typename Aggregator>
graph_conv_layer<Aggregator>::graph_conv_layer(int id, int nv, int din, int dout, 
                 LearningGraph *g, bool act, bool concat, float lr, float feat_drop, float score_drop) :
    level_(id), num_samples(nv), dim_in(din), dim_out(dout), graph(g), is_act(act),
    is_bias(false), use_concat(concat), feat_dropout_rate(feat_drop), score_dropout_rate(score_drop) {
  auto x = num_samples;
  auto y = dim_in;
  auto z = dim_out;
  //std::cout << "Init graph_conv layer, nv = " << x << ", dim_in = " << dim_in << ", dim_out = " << dim_out << "\n";
  auto init_range = sqrt(6.0 / (y + z));
  float_malloc_device(y * z, d_W_neigh);
  rng_uniform_gpu(y * z, -init_range, init_range, d_W_neigh);
  float_malloc_device(y * z, d_W_neigh_grad);
  init_const_gpu(y * z, 0.0, d_W_neigh_grad);
  if (concat) {
    float_malloc_device(y * z, d_W_self);
    rng_uniform_gpu(y * z, -init_range, init_range, d_W_self);
    float_malloc_device(y * z, d_W_self_grad);
    init_const_gpu(y * z, 0.0, d_W_self_grad);
  }

  float_malloc_device(x * y, d_in_temp);
  float_malloc_device(x * z, d_out_temp);
  init_const_gpu(x * y, 0.0, d_in_temp);
  init_const_gpu(x * z, 0.0, d_out_temp);
  if (y <= z) {
    float_malloc_device(x * y, d_in_temp1);
    init_const_gpu(x * y, 0.0, d_in_temp1);
  }

  if (level_ > 0) {
    float_malloc_device(x * y, feat_in);
    init_const_gpu(x * y, 0.0, feat_in);
  }
  float_malloc_device(x * z, grad_in);
  init_const_gpu(x * z, 0.0, grad_in);

  assert(feat_dropout_rate>=0. && feat_dropout_rate<1.);
  assert(score_dropout_rate>=0. && score_dropout_rate<1.);
  feat_scale = 1. / (1. - feat_dropout_rate);
  if (feat_dropout_rate) uint8_malloc_device(x * y, dropout_mask);

  if (is_bias) {
    float_malloc_device(z, d_bias);
    float_malloc_device(z, d_bias_grad);
    init_const_gpu(z, 0.0, d_bias);
    init_const_gpu(z, 0.0, d_bias_grad);
  }

  optm = new adam(lr);
}

template <typename Aggregator>
void graph_conv_layer<Aggregator>::update_dim_size(size_t x) {
  if (x > num_samples) {
    int y = dim_in;
    int z = dim_out;
    if (d_in_temp) float_free_device(d_in_temp);
    if (d_out_temp) float_free_device(d_out_temp);
    float_malloc_device(x * y, d_in_temp);
    float_malloc_device(x * z, d_out_temp);
    if (y <= z) {
      if (d_in_temp1) float_free_device(d_in_temp1);
      float_malloc_device(x * y, d_in_temp1);
    }
    if (level_ > 0) {
      if (feat_in) float_free_device(feat_in);
      float_malloc_device(x * y, feat_in);
    }
    if (grad_in) float_free_device(grad_in);
    float_malloc_device(x * z, grad_in);
    if (feat_dropout_rate) {
      if (dropout_mask) uint8_free_device(dropout_mask);
      uint8_malloc_device(x * y, dropout_mask);
    }
  }
  num_samples = x;
  if (dim_in < dim_out) aggr.init(dim_in, x);
  else aggr.init(dim_out, x);
}

template class graph_conv_layer<GCN_Aggregator>;
template class graph_conv_layer<GAT_Aggregator>;
template class graph_conv_layer<SAGE_Aggregator>;
template class graph_conv_layer<GGNN_Aggregator>;

