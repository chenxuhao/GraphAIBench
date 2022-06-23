#include "l2norm_layer.h"
#include "math_functions.hh"

l2norm_layer::l2norm_layer(int nv, int len) :
   num_samples(nv), dim(len) {
#ifdef ENABLE_GPU
  float_malloc_device(nv*dim, feat_in);
  float_malloc_device(nv*dim, grad_in);
  init_const_gpu(nv*dim, 0.0, feat_in);
  init_const_gpu(nv*dim, 0.0, grad_in);
#else
  feat_in = new float[nv*dim];
  grad_in = new float[nv*dim];
  std::fill(feat_in, feat_in+nv*dim, 0);
  std::fill(grad_in, grad_in+nv*dim, 0);
#endif
}

void l2norm_layer::forward(float* feat_out) {
  double t1 = omp_get_wtime();
#ifdef ENABLE_GPU
  l2norm(num_samples, dim, feat_in, feat_out);
#else
  #pragma omp parallel for
  for (int i = 0; i < num_samples; i++) {
    float sum = 0;
    for (int j = 0; j < dim; j++)
      sum += feat_in[i*dim+j] * feat_in[i*dim+j];
    sum = sum < 1.0e-12 ? 1.0e-12 : sum;
    sum = sqrt(sum);
    assert(sum!=0.);
    for (int j = 0; j < dim; j++)
      feat_out[i*dim+j] = feat_in[i*dim+j]/sum;
  }
#endif
  double t2 = omp_get_wtime();
  time_ops[OP_NORM] += t2-t1;
}

void l2norm_layer::backward(float* grad_out) {
  double t1 = omp_get_wtime();
#ifdef ENABLE_GPU
  d_l2norm(num_samples, dim, feat_in, grad_in, grad_out);
#else
  #pragma omp parallel for
  for (int i = 0; i < num_samples; i++) {
    float coef0_axis0 = 0, coef1_axis0 = 0;
    float sum_x2 = 0;
    for (int j = 0; j < dim; j++) {
      sum_x2 += powf(feat_in[i*dim+j], 2);
      coef0_axis0 -= feat_in[i*dim+j] * grad_in[i*dim+j];
    }
    sum_x2 = sum_x2 < 1.0e-12 ? 1.0e-12 : sum_x2;
    assert(sum_x2!=0.);
    coef1_axis0 = powf(sum_x2, -1.5);
    for (int j = 0; j < dim; j++) {
      grad_out[i*dim+j] = feat_in[i*dim+j] * coef0_axis0 * coef1_axis0 +
                          grad_in[i*dim+j] * sum_x2 * coef1_axis0;
    }
  }
#endif
  double t2 = omp_get_wtime();
  time_ops[OP_NORM] += t2-t1;
}

void l2norm_layer::update_dim_size(int x) {
  if (x > num_samples) {
#ifdef ENABLE_GPU
    if (feat_in) float_free_device(feat_in);
    float_malloc_device(x*dim, feat_in);
    init_const_gpu(x*dim, 0.0, feat_in);
#else
    if (feat_in) delete[] feat_in;
    feat_in = new float[x*dim];
    std::fill(feat_in, feat_in+x*dim, 0);
    //if (grad_in) delete[] grad_in;
    //grad_in = new float[x*dim_out];
    //std::fill(grad_in, grad_in+x*dim_out, 0);
#endif
  }
  num_samples = x;
}
