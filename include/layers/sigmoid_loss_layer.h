#pragma once
#include "loss_layer.h"

// the output layer (multi-class vertex classification)
class sigmoid_loss_layer : public loss_layer {
  public:
    sigmoid_loss_layer() { }
    sigmoid_loss_layer(int nv, int n_cls) : loss_layer(nv, n_cls, NULL) { }
    sigmoid_loss_layer(int nv, int n_cls, label_t* ptr) : loss_layer(nv, n_cls, ptr) { }
    virtual void forward(size_t begin, size_t end, mask_t* masks);
    virtual void backward(size_t begin, size_t end, mask_t* masks, float* grad_out);
    virtual acc_t get_prediction_loss(size_t begin, size_t end, size_t count, mask_t* masks);
};
