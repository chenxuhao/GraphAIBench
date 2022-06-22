#pragma once
#include "global.h"

// the output layer
class loss_layer {
  public:
    loss_layer();
    loss_layer(int nv, int n_cls);
    loss_layer(int nv, int n_cls, label_t* ptr);
    float* get_feat_in() { return feat_in; }
    float* get_feat_out() { return feat_out; }
    virtual void forward(size_t begin, size_t end, mask_t* masks) {}
    virtual void backward(size_t begin, size_t end, mask_t* masks, float* grad_out) {}
    void set_labels_ptr(label_t* ptr) { labels = ptr; }
    virtual acc_t get_prediction_loss(size_t begin, size_t end, size_t count, mask_t* masks) { return 0; }
    void set_netphase(net_phase phase) { phase_ = phase; }
    void update_dim_size(int sz);
    void print_layer_info() {
      std::cout << "Output Layer with " << num_samples << " samples and " << num_cls << " classes\n";
    }

  protected:
    int num_samples;
    int num_cls;
    net_phase phase_;          // in which phase: train, val or test
    float* feat_in;            // there is no grad_in for this final layer
    float* feat_out;
    std::vector<acc_t> losses; // loss for each sample
    label_t* labels;
    acc_t* d_losses;           // losses on device, e.g. GPU
};

