#include "optimizer.h"
#include "math_functions.hh"

void adagrad::update(const vec_t& dW, vec_t& W) {
  vec_t& g = get<0>(W);
  #pragma omp parallel for
  for (size_t i = 0; i < W.size(); i++) {
    g[i] += dW[i] * dW[i];
    W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
  }
}

void RMSprop::update(const vec_t& dW, vec_t& W) {
  vec_t& g = get<0>(W);
  #pragma omp parallel for
  for (size_t i = 0; i < W.size(); i++) {
    g[i] = mu * g[i] + (1 - mu) * dW[i] * dW[i];
    W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);
  }
}

void adam::update(const vec_t& dW, vec_t& W) {
  vec_t& mt = get<0>(W);
  vec_t& vt = get<1>(W);
  #pragma omp parallel for
  for (size_t i = 0; i < W.size(); i++) {
    mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
    vt[i] = b2 * vt[i] + (float_t(1) - b2) * dW[i] * dW[i];
    // L2 norm based update rule
    W[i] -= alpha * (mt[i] / (float_t(1) - b1_t)) /
      std::sqrt((vt[i] / (float_t(1) - b2_t)) + eps);
  }
  b1_t *= b1;
  b2_t *= b2;
}

void adamax::update(const vec_t& dW, vec_t& W) {
  vec_t& mt = get<0>(W);
  vec_t& ut = get<1>(W);
  #pragma omp parallel for
  for (size_t i = 0; i < W.size(); i++) {
    mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
    ut[i] = std::max(b2 * ut[i], std::abs(dW[i]));
    // Lp norm based update rule
    W[i] -= (alpha / (1.0 - b1_t)) * (mt[i] / (ut[i] + eps));
  }
  b1_t *= b1;
}

void gradient_descent::update(const vec_t& dW, vec_t& W) {
  #pragma omp parallel for
  for (size_t i = 0; i < W.size(); i++)
    W[i] = W[i] - alpha * (dW[i] + lambda * W[i]);
}

void momentum::update(const vec_t& dW, vec_t& W) {
  vec_t& dWprev = get<0>(W);
  #pragma omp parallel for
  for (size_t i = 0; i < W.size(); i++) {
    float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
    W[i] += V;
    dWprev[i] = V;
  }
}

void nesterov_momentum::update(const vec_t& dW, vec_t& W) {
  vec_t& dWprev = get<0>(W);
  #pragma omp parallel for
  for (size_t i = 0; i < W.size(); i++) {
    float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
    W[i] += (-mu) * dWprev[i] + (1 + mu) * V;
    dWprev[i] = V;
  }
}

