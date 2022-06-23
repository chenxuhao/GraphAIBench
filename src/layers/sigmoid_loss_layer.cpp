#include "math_functions.hh"
#include "sigmoid_loss_layer.h"

void sigmoid_loss_layer::forward(size_t begin, size_t end, mask_t* masks) {
  //std::cout << "Loss layer forward: [" << num_samples << " x " << num_cls << "]\n";
  double t1 = omp_get_wtime();
  #pragma omp parallel for
  for (auto i = begin; i < end; i++) {
    if (masks == NULL || masks[i] == 1) {
      auto idx = num_cls * i;
      sigmoid(num_cls, &feat_in[idx], &feat_out[idx]);
      losses[i] = sigmoid_cross_entropy(num_cls, &labels[idx], &feat_in[idx]);
    }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_LOSS] += t2 - t1;
}

void sigmoid_loss_layer::backward(size_t begin, size_t end, mask_t* masks, float* grad_out) {
  double t1 = omp_get_wtime();
  #pragma omp parallel for
  for (auto i = begin; i < end; i++) {
    if (masks == NULL || masks[i] == 1) {
      auto idx = num_cls * i;
      float *grad = &grad_out[idx];
      for (int j = 0; j < num_cls; j++) {
        auto pred = feat_out[idx+j];
        grad[j] = (pred - float(labels[idx+j])) / (end - begin); // TODO: use count, not end-begin
      }
    }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_LOSS] += t2 - t1;
}

acc_t sigmoid_loss_layer::get_prediction_loss(size_t begin, size_t end, size_t count, mask_t* masks) {
  acc_t total_loss = 0.0;
  size_t valid_sample_count = 0;
  #pragma omp parallel for reduction(+:total_loss,valid_sample_count)
  for (auto i = begin; i < end; i++) {
    if (masks == NULL || masks[i] == 1) {
      total_loss += losses[i];
      valid_sample_count += 1;
    }
  }
  //if (valid_sample_count != count)
  //  std::cout << "valid_sample_count = " << valid_sample_count << ", count = " << count << "\n";
  assert(valid_sample_count == count);
  if (valid_sample_count > 0) {
    return total_loss / (acc_t)valid_sample_count;
  } else return 0;
}
