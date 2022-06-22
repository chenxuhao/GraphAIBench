#include "loss_layer.h"
#include "math_functions.h"

loss_layer::loss_layer() {
  loss_layer(0, 1, NULL);
}

loss_layer::loss_layer(int nv, int ncls) {
  loss_layer(nv, ncls, NULL);
}

loss_layer::loss_layer(int nv, int ncls, label_t* ptr) :
    num_samples(nv), num_cls(ncls), labels(ptr) {
  //std::cout << "Output Layer allocating memory: [" << nv << " x " << num_cls << "]\n";
#ifdef ENABLE_GPU
  float_malloc_device(nv*num_cls, feat_in);
  float_malloc_device(nv*num_cls, feat_out);
  init_const_gpu(nv*num_cls, 0.0, feat_in);
  init_const_gpu(nv*num_cls, 0.0, feat_out);
  float_malloc_device(nv, d_losses);
  init_const_gpu(nv, 0.0, d_losses);
#else
  feat_in = new float[nv*num_cls];
  feat_out = new float[nv*num_cls];
  std::fill(feat_in, feat_in+nv*num_cls, 0);
  std::fill(feat_out, feat_out+nv*num_cls, 0);
  losses.resize(nv);
  std::fill(losses.begin(), losses.end(), 0);
#endif
}

void loss_layer::update_dim_size(int x) {
  if (x > num_samples) {
#ifdef ENABLE_GPU
    if (feat_in) float_free_device(feat_in);
    float_malloc_device(x*num_cls, feat_in);
    init_const_gpu(x*num_cls, 0.0, feat_in);
    if (feat_out) float_free_device(feat_out);
    float_malloc_device(x*num_cls, feat_out);
    init_const_gpu(x*num_cls, 0.0, feat_out);
    if (d_losses) float_free_device(d_losses);
    float_malloc_device(x, d_losses);
    init_const_gpu(x, 0.0, d_losses);
#else
    if (feat_in) delete[] feat_in;
    feat_in = new float[x*num_cls];
    if (feat_out) delete[] feat_out;
    feat_out = new float[x*num_cls];
    std::fill(feat_in, feat_in+x*num_cls, 0);
    std::fill(feat_out, feat_out+x*num_cls, 0);
    losses.resize(x);
    std::fill(losses.begin(), losses.end(), 0);
#endif
  }
  num_samples = x;
}
