#include "math_functions.h"
#include "softmax_loss_layer.h"

void softmax_loss_layer::forward(size_t begin, size_t end, mask_t* masks) {
  double t1 = omp_get_wtime();
  #pragma omp parallel
  {
  vec_t ground_truth(num_cls);
  #pragma omp for
  for (auto i = begin; i < end; i++) {
    if (masks == NULL || masks[i] == 1) {
      softmax(num_cls, &feat_in[num_cls*i], &feat_out[num_cls*i]);
      for (int i = 0; i < num_cls; i++) ground_truth[i] = 0.;
      ground_truth[labels[i]] = 1.0;
      losses[i] = cross_entropy(num_cls, &ground_truth[0], &feat_out[num_cls*i]);
    }
  }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_LOSS] += t2 - t1;
}

void softmax_loss_layer::backward(size_t begin, size_t end, mask_t* masks, float* grad_out) {
  double t1 = omp_get_wtime();
  #pragma omp parallel for
  for (auto i = begin; i < end; i++) {
    if (masks == NULL || masks[i] == 1) {
      auto idx = num_cls * i;
      for (int j = 0; j < num_cls; j++) {
        auto pred = feat_out[idx+j];
        grad_out[idx+j] = (pred - (labels[i]==j?1.0:0.0)) / (end - begin); // TODO: use count, not end-begin
      }
    }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_LOSS] += t2 - t1;
}

acc_t softmax_loss_layer::get_prediction_loss(size_t begin, size_t end, size_t count, mask_t* masks) {
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
  acc_t avg_loss = 0;
  if (valid_sample_count > 0) avg_loss = total_loss / (acc_t)valid_sample_count;
  return avg_loss;
}

