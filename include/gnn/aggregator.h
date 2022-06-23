#pragma once
#include "lgraph.h"
#include "optimizer.h"
#include "math_functions.hh"
//typedef LearningGraph Graph;

//template<bool learnable=false>
class aggregator {
public:
  aggregator() : n(0), length(0), temp(NULL) {}
  //void aggregate(Graph& g, const float* in, float* out);
  //void d_aggregate(Graph& g, const float* in, float* out);
  void set_vlen(int vlen) { length = vlen; }

protected:
  int n;
  int length;    // feature vector length
  float* temp;   // for calling cusparse csrmm/spmm
};

class GCN_Aggregator : public aggregator {
public:
  GCN_Aggregator() : aggregator() {}
  void init(int length, int nv, int ne = 0, float lr = 0.01, float drop_rate = 0.);
  void aggregate(int len, Graph& g, const float* in, float* out);
  void d_aggregate(int len, Graph& g, const float* feat_in, const float* grad_in, float* grad_out);
protected:
  std::vector<std::vector<vec_t>> partial_sums;
  vec_t& get_partial_feats(int bid, index_t vid);
  void update_all(int len, Graph& g, const float* in, float* out);
  void update_all_blocked(int len, Graph& g, const float* in, float* out);
};

class SAGE_Aggregator : public aggregator {
public:
  SAGE_Aggregator() : aggregator() {}
  void init(int length, int nv, int ne = 0, float lr = 0.01, float drop_rate = 0.);
  void aggregate(int len, Graph& g, const float* in, float* out);
  void d_aggregate(int len, Graph& g, const float* feat_in, const float* grad_in, float* grad_out);
};

class GAT_Aggregator : public aggregator {
public:
  GAT_Aggregator() : aggregator() {}
  void init(int length, int nv, int ne = 0, float lr = 0.01, float drop_rate = 0.);
  void aggregate(int len, Graph& g, const float* in, float* out);
  void d_aggregate(int len, Graph& g, const float* feat_in, const float* grad_in, float* grad_out);
  void update_weights(optimizer *opt); // update alpha

private:
  // GAT parameters
  float epsilon;           // LeakyReLU angle of negative slope. Default: `` 0.2``.
  float attn_drop;         // Dropout rate on attention weight. Default: ``0``.
  float attn_scale;        // Scale rate for attention dropout. Default: ``1``.
  vec_t alpha_l;           // parameters to learn (H x 1), only used for GAT
  vec_t alpha_r;           // parameters to learn (H x 1), only used for GAT
  vec_t alpha_lgrad;       // gradients for updating alpha (GAT only)
  vec_t alpha_rgrad;       // gradients for updating alpha (GAT only)
  vec_t scores;            // un-normalized scores
  vec_t temp_scores;       // un-normalized scores
  vec_t scores_grad;       // gradients of un-normalized scores
  vec_t norm_scores;       // normalized scores
  vec_t norm_scores_grad;  // gradients of normalized scores
  vec_t trans_norm_scores; // transposed normalized scores
  optimizer *alpha_opt;    // optimizer for alpha
  std::vector<mask_t> attn_masks;  // masks for attention dropout.

  // GPU related
  float* d_alpha_l;           // parameters to learn (H x 1), only used for GAT
  float* d_alpha_r;           // parameters to learn (H x 1), only used for GAT
  float* d_alpha_lgrad;       // gradients for updating alpha (GAT only)
  float* d_alpha_rgrad;       // gradients for updating alpha (GAT only)
  float* d_scores;            // un-normalized scores
  float* d_temp_scores;       // un-normalized scores
  float* d_scores_grad;       // gradients of un-normalized scores
  float* d_norm_scores;       // normalized scores
  float* d_norm_scores_grad;  // gradients of normalized scores
  float* d_trans_norm_scores; // transposed normalized scores
  float* d_rands;             // random numbers between 0 and 1
  mask_t* d_attn_masks;       // masks for attention dropout.
};

