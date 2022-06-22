#include "math_functions.h"
#include "sigmoid_loss_layer.h"

void sigmoid_loss_layer::forward(size_t begin, size_t end, mask_t* masks) {
  double t1 = omp_get_wtime();
  init_const_gpu(num_samples, 0.0, d_losses);
  sigmoid_cross_entropy_gpu(num_cls, begin, end, feat_in, masks, labels, d_losses, feat_out);
  double t2 = omp_get_wtime();
  time_ops[OP_LOSS] += t2 - t1;
}

void sigmoid_loss_layer::backward(size_t begin, size_t end, mask_t* masks, float* grad_out) {
  double t1 = omp_get_wtime();
  //init_const_gpu(num_samples*num_cls, 0.0, grad_out);
  d_sigmoid_cross_entropy_gpu(num_cls, begin, end, masks, labels, feat_out, grad_out);
  double t2 = omp_get_wtime();
  time_ops[OP_LOSS] += t2 - t1;
}

acc_t sigmoid_loss_layer::get_prediction_loss(size_t begin, size_t end, size_t count, mask_t* masks) {
  assert(end > begin);
  return masked_avg_loss_gpu(begin, end, count, masks, d_losses);
}

