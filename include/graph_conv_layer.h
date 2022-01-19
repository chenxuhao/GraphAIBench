#pragma once
#include "lgraph.h"
#include "optimizer.h"
#include "aggregator.h"

template <typename Aggregator>
class graph_conv_layer {
  public:
    graph_conv_layer(int id, int nv, int din, int dout, Graph *g, bool act,
                     bool concat, float lr, float feat_drop, float score_drop);
    float* get_feat_in() { return feat_in; }
    float* get_grad_in() { return grad_in; }
    void set_feat_in(float* ptr) { feat_in = ptr; }
    void set_graph_ptr(Graph* ptr) { graph = ptr; }
    void set_netphase(net_phase phase) { phase_ = phase; }
    void print_layer_info() {
      std::cout << "GraphConv Layer " << level_ << " with " << num_samples
                << " samples, dims: [" << dim_in << " x " << dim_out << "]\n";
    }
    void update_dim_size(size_t sz); // update number of vertices; useful for subgraph sampling

  protected:
    int level_;
    int num_samples;              // number of vertices
    int dim_in;                   // input feature vector length
    int dim_out;                  // output feature vector length
    Graph *graph;                 // input graph
    bool is_act;                  // whether to use activation function at the end
    bool is_bias;                 // whether to add bias before relu
    bool use_concat;              // concat self and neighbors' features
    float feat_dropout_rate;      // dropout rate for features
    float score_dropout_rate;     // dropout rate for scores
    float feat_scale;             // scale rate used for feature dropout
    net_phase phase_;             // in which phase: train, val or test

    vec_t in_temp;                // intermediate features, [nv*dim_in]
    vec_t in_temp1;               // intermediate features, [nv*dim_in]
    vec_t out_temp;               // intermediate features, [nv*dim_out]
    vec_t W_neigh;                // layer parameters: weight for neighbor aggregation, [dim_in*dim_out]
    vec_t W_neigh_grad;           // gradients for layer parameters, [dim_in*dim_out]
    vec_t W_self;                 // layer parameters, [dim_in*dim_out]
    vec_t W_self_grad;            // gradients for layer parameters, [dim_in*dim_out]
    vec_t bias;                   // layer parameters: bias [dim_out]
    vec_t bias_grad;              // gradients for bias [dim_out]
    float* feat_in;               // input features. reuse in backprop. [nv*dim_in]
    float* grad_in;               // input gradients, [nv*dim_out]
    float* d_in_temp;             // intermediate features on GPU, [nv*dim_in]
    float* d_in_temp1;            // intermediate features on GPU, [nv*dim_in]
    float* d_out_temp;            // intermediate features on GPU, [nv*dim_out]
    float* d_W_neigh;             // weight for neighbor aggregation on GPU
    float* d_W_neigh_grad;        // weight gradients on GPU
    float* d_W_self;              // weight for self on GPU
    float* d_W_self_grad;         // weight gradients on GPU
    float* d_bias;                // bias on GPU
    float* d_bias_grad;           // bias gradients on GPU
    mask_t* dropout_mask;         // masks to record which features are dropped out
    optimizer *optm;              // optimizer for learnable weights update
    Aggregator aggr;              // aggregator
};

class GCN_layer : public graph_conv_layer<GCN_Aggregator> {
  public:
    GCN_layer(int id, int nv, int din, int dout, Graph *g, bool act, 
              float lr, float feat_drop_rate, float score_drop_rate) :
              graph_conv_layer(id, nv, din, dout, g, act, false, lr, feat_drop_rate, score_drop_rate) {
      if (dim_in < dim_out) aggr.init(dim_in, nv);
      else aggr.init(dim_out, nv);
    }
    void forward(float* feat_out);
    void backward(float* feat_out, float* grad_out);
    void update_weight(optimizer* opt);
};

class SAGE_layer : public graph_conv_layer<SAGE_Aggregator> {
  public:
    SAGE_layer(int id, int nv, int din, int dout, Graph *g, bool act, 
               float lr, float feat_drop_rate, float score_drop_rate) :
               graph_conv_layer(id, nv, din, dout, g, act, true, lr, feat_drop_rate, score_drop_rate) {
      if (dim_in < dim_out) aggr.init(dim_in, nv);
      else aggr.init(dim_out, nv);
    }
    void forward(float* feat_out);
    void backward(float* feat_out, float* grad_out);
    void update_weight(optimizer* opt);
};

class GAT_layer : public graph_conv_layer<GAT_Aggregator> {
  public:
    GAT_layer(int id, int nv, int din, int dout, Graph *g, bool act,
              float lr, float feat_drop_rate, float score_drop_rate) :
              graph_conv_layer(id, nv, din, dout, g, act, false, lr, feat_drop_rate, score_drop_rate) {
      aggr.init(dim_out, nv, g->sizeEdges(), lr, score_drop_rate);
    }
    void forward(float* feat_out);
    void backward(float* feat_out, float* grad_out);
    void update_weight(optimizer* opt);
};

