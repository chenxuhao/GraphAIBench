#include "math_functions.hh"
#include "graph_conv_layer.h"

template <typename Aggregator>
graph_conv_layer<Aggregator>::graph_conv_layer(int id, int nv, int din, int dout, Graph *g,
                      bool act, bool concat, float lr, float feat_drop, float score_drop) :
    level_(id), num_samples(nv), dim_in(din), dim_out(dout), graph(g), is_act(act),
    is_bias(false), use_concat(concat), feat_dropout_rate(feat_drop), score_dropout_rate(score_drop) {
  //std::cout << "GCN Layer " << level_ << " allocating memory: [" << num_samples << " x " << dim_in << "]\n";
  int z = dim_out;
  W_neigh.resize(dim_in*z);
  W_neigh_grad.resize(dim_in*z);
  init_glorot(dim_in, z, W_neigh, 1); // init weight
  std::fill(W_neigh_grad.begin(), W_neigh_grad.end(), 0);
  if (concat) {
    W_self.resize(dim_in*z);
    W_self_grad.resize(dim_in*z);
    init_glorot(dim_in, z, W_self, 2); // init weight
    std::fill(W_self_grad.begin(), W_self_grad.end(), 0);
  }

  in_temp.resize(num_samples*dim_in);
  out_temp.resize(num_samples*dim_out);
  std::fill(in_temp.begin(), in_temp.end(), 0);
  std::fill(out_temp.begin(), out_temp.end(), 0);
  if (dim_in <= dim_out) {
    in_temp1.resize(num_samples*dim_in);
    std::fill(in_temp1.begin(), in_temp1.end(), 0);
  }

  if (level_ > 0) {
    feat_in = new float[num_samples*dim_in];
    std::fill(feat_in, feat_in+num_samples*dim_in, 0);
  }
  grad_in = new float[num_samples*dim_out];
  std::fill(grad_in, grad_in+num_samples*dim_out, 0);

  assert(feat_dropout_rate>=0. && feat_dropout_rate<1.);
  assert(score_dropout_rate>=0. && score_dropout_rate<1.);
  feat_scale = 1. / (1. - feat_dropout_rate);
  if (feat_dropout_rate > 0.) dropout_mask = new mask_t[num_samples*dim_in];

  if (is_bias) {
    bias.resize(dim_out);
    bias_grad.resize(dim_out);
    std::fill(bias.begin(), bias.end(), 0);
    std::fill(bias_grad.begin(), bias_grad.end(), 0);
  }
 
  optm = new adam(lr);
}

template <typename Aggregator>
void graph_conv_layer<Aggregator>::update_dim_size(size_t x) {
  num_samples = x;
}

template class graph_conv_layer<GCN_Aggregator>;
template class graph_conv_layer<GAT_Aggregator>;
template class graph_conv_layer<SAGE_Aggregator>;

