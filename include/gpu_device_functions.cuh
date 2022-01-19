#pragma once
#include "global.h"

__inline__ __device__ void softmax(int n, const float* input, float* output) {
  float max = input[0];
  for (int i = 1; i < n; i++) if (input[i] > max) max = input[i];
  float denominator = 0.0;
  for (int i = 0; i < n; i++) {
    output[i] = expf(input[i] - max);
    denominator += output[i];
  }
  for (int i = 0; i < n; i++) output[i] /= denominator;
}

__inline__ __device__ void d_softmax(int n, const float* p, const float* dp, float* dy) {
  for (int i = 0; i < n; i++) {
    dy[i] = 0;
    for (int j = 0; j < n; j++) {
      float df = (j == i) ? p[i] * (1.0 - p[i]) : -p[j] * p[i];
      dy[i] += df * dp[j];
    }
  }
}

__inline__ __device__ void dropout(int n, float scale, float threshold,
                        float* rands, float* in, mask_t* masks, float* out) {
  for (int i = 0; i < n; i++) {
    masks[i] = rands[i] > threshold ? 1 : 0;
    out[i]   = in[i] * masks[i] * scale;
  }
}

__inline__ __device__ void d_dropout(index_t n, float scale, const float* in, const mask_t* masks, float* out) {
  for (index_t i = 0; i < n; i ++) { out[i] = in[i] * masks[i] * scale; }
}

__inline__ __device__ float dot_product(int n, const float* a, const float* b) {
  float sum = 0.;
  for (int i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

// TODO: use warp
__inline__ __device__ void softmax_device(int n, const float* input, float* output) {
  float max = input[0];
  for (int i = 1; i < n; i++)
    if (input[i] > max)
      max = input[i];
  float denominator = 0.0;
  for (int i = 0; i < n; i++) {
    output[i] = expf(input[i] - max);
    denominator += output[i];
    //if (output[i] < 0.0) printf("in[%d]=%f, out[%d]=%f\n", i, input[i], i, output[i]);
  }
  assert(denominator != 0.0);
  for (int i = 0; i < n; i++) {
    output[i] /= denominator;
  }
}

__inline__ __device__ void sigmoid_device(int n, const float* in, float* out) {
  for (int i = 0; i < n; i++)
    out[i] = 1. / (1. + expf(-in[i]));
}

__inline__ __device__ void cross_entropy_device(int n, const label_t idx, const float* p, float& loss) {
  loss -= p[idx] == 0.0 ? logf(float(1e-10)) : logf(p[idx]);
}

// y: ground truth
// p: predictions
__inline__ __device__ void cross_entropy_multi_device(int n, const label_t* y,
                                           const float_t* p, float_t& loss) {
  for (int i = 0; i < n; i++) {
    if (y[i] == 0)
      continue;
    if (p[i] == float_t(0))
      loss -= logf(float_t(1e-10)); // avoid nan exception
    else
      loss -= logf(p[i]);
  }
}

__inline__ __device__ void d_cross_entropy_device(int n, const label_t idx,
                                       const float_t* p, float_t* d) {
  for (int i = 0; i < n; i++) {
    if (i == (int)idx)
      d[i] = -1.0 / (p[i] + 1e-10);
    else
      d[i] = 0.0;
  }
}

__inline__ __device__ void d_softmax_device(int n, const float* p, const float* dp, float* dy) {
  for (int i = 0; i < n; i++) {
    dy[i] = 0;
    for (int j = 0; j < n; j++) {
      float_t df = (j == i) ? p[i] * (1.0 - p[i]) : -p[j] * p[i];
      dy[i] += df * dp[j];
    }
  }
}

// the arguments of the maxima
__inline__ __device__ int argmax_device(const int n, const float_t* x) {
  float_t max = x[0];
  int max_ind = 0;
  for (int i = 1; i < n; i++) {
    if (x[i] > max) {
      max_ind = i;
      max     = x[i];
    }
  }
  return max_ind;
}

