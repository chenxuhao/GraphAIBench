#pragma once
#include "loss_layer.h"

// the output layer (single-class vertex classification)
class softmax_loss_layer : public loss_layer {
  public:
    softmax_loss_layer() { }
    softmax_loss_layer(int nv, int n_cls) : loss_layer(nv, n_cls, NULL) { }
    softmax_loss_layer(int nv, int n_cls, label_t* ptr) : loss_layer(nv, n_cls, ptr) { }
    virtual void forward(size_t begin, size_t end, mask_t* masks);
    virtual void backward(size_t begin, size_t end, mask_t* masks, float* grad_out);
    virtual acc_t get_prediction_loss(size_t begin, size_t end, size_t count, mask_t* masks);

  private:
    //std::vector<acc_t> losses; // loss for each sample
};

