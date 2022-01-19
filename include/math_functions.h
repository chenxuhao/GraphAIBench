#pragma once
#include "global.h"

#ifdef USE_MKL
#include <mkl.h>
inline void* _malloc(size_t size) {return mkl_malloc(size,64);}
inline void _free(void* ptr) {return mkl_free(ptr);}
#else
#include <cblas.h>
inline void* _malloc(size_t size) {return malloc(size);}
inline void _free(void* ptr) {return free(ptr);}
#endif

void init_glorot(size_t dim_x, size_t dim_y, vec_t &weight, unsigned seed);

// symmetric sparse matrix transpose
void symmetric_csr_transpose(int N, int nnz, int* A_idx_ptr, int* A_nnz_idx,
                             float* A_nonzeros, float* &B_nonzeros);
 
// get accuracy for single-class problems
float masked_accuracy_single(int begin, int end, int count, int num_classes,
                             mask_t* masks, float* preds, label_t* ground_truth);
 
// get accuracy for multi-class problems
float masked_accuracy_multi(int begin, int end, int count, int num_classes,
                            mask_t* masks, float* preds, label_t* ground_truth);

void matmul_naive(const size_t x, const size_t y, const size_t z,
                  const float* A, const float* B, float* C);
void matmul(const size_t x, const size_t y, const size_t z,
            const float_t* A, const float_t* B, float* C, 
            bool transA = false, bool transB = false, bool accum = false);

void l2norm(int n, int dim, const float* in, float* out);
void d_l2norm(int n, int dim, const float* feat_in, const float* grad_in, float* grad_out);
void bias_mv(int n, int len, float* x, float* b);
void reduce_sum(int n, int len, float* x, float* a);
void reduce_sum(int n, int len, float* x, vec_t &a);

// single-precision dense matrix multiply
void sgemm_cpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const float alpha,
               const float* A, const float* B, const float beta, float* C);

void csr2csc(int nrows, int ncols, int nnz,
             const float* values, const int* rowptr, const int* colidx,
             float* valuesT, int* rowptrT, int* colidxT);
 
// single-precision sparse matrix dense matrix multiply, C = A * B, A is sparse
void csrmm_cpu(const int M, const int N, const int K, const int nnz,
               const float alpha, float* A_nonzeros, int* A_idx_ptr,
               int* A_nonzero_idx, const float* B, const float beta, float* C, bool transA = false);

void spmm(size_t x, size_t y, size_t z, size_t nnz,
          float* A_nonzeros, int* A_idx_ptr, int* A_nnz_idx,
          const float* B, float* C, float* temp = NULL, 
          bool transA = false, bool transB = false, bool accum = false);
 
// matrix-vector multiply
void mvmul(const CBLAS_TRANSPOSE TransA, const int M, const int N,
           const float alpha, const float* A, const float* x, const float beta, float* y);

//! add 2 arrays for n elements
void vadd_cpu(int n, const float* a, const float* b, float* out);
void scaled_vadd_cpu(int n, const float a, const float* x, const float* y, float* z);

// atomic y[i] += x[i]
void atomic_vreduce_cpu(size_t n, const float* x, float* y);

//! multiply n elements of vector by scalar
void scal(size_t n, const float alpha, float* x);
void scale(int n, const float alpha, const float* x, float* y);
void mul_scalar(size_t n, const float alpha, const float* x, float* y);

//! do dot product of 2 vectors
float dot(int n, const float* x, const float* y);

// concatenation of two vectors into one
void concat(size_t n, const float* x, const float* y, float* z);

// SAXPY stands for Single-precision A*X Plus Y"
void axpy(size_t n, const float a, float* x, float* y);

// Returns the index of the maximum value
int argmax(int num, const float* arr);

//! clear n elements of a vector
void clear_cpu(int n, float* in);

//! copy vector from in -> out; first len elements
void copy_cpu(size_t len, const float* in, float* out);

// dropout functions randomly remove weights
void dropout_cpu(size_t n, size_t m, float scale, float dropout_rate,
                 const float* in, mask_t* mask, float* out);

// dropout derivative: use existing dropouts in masks instead of generating them;
void d_dropout_cpu(size_t n, size_t m, float scale, const float* in,
                   const mask_t* mask, float* out);

void leaky_relu(float epsilon, float in, float &out);
void d_leaky_relu(float epsilon, float in, float data, float &out);
void relu_cpu(size_t n, const float* in, float* out);
void d_relu_cpu(size_t n, const float* in, const float* data, float* out);
void leaky_relu_cpu(size_t n, float epsilon, const float* in, float* out);
void d_leaky_relu_cpu(size_t n, float epsilon, const float* in, const float* data, float* out);
void softmax(size_t n, const float* in, float* out);
void d_softmax(int n, const float* p, const float* dp, float* dy);
void sigmoid(size_t n, const float* in, float* out);
void d_sigmoid(size_t n, const float*, const float* p, float* dy, const float* dp);
float cross_entropy(size_t n, const float* y, const float* p);
void d_cross_entropy(size_t n, const float* y, const float* p, float* d);
float sigmoid_cross_entropy(size_t n, const label_t* y, const float* p);

// use sigmoid instead of softmax for multi-class datasets, e.g. ppi, yelp and amazon 
//inline float sigmoid_func(float x) { return 0.5 * tanh(0.5 * x) + 0.5; }
inline float sigmoid_func(float x) { return 1. / (1. + expf(-x)); }

// GPU operators
bool isnan_gpu(int n, const float_t* array); // does array contain any 'nan' element
void init_const_gpu(int n, float_t value, float_t* array);
void copy_gpu(int len, const float_t* in, float_t* out);
void vadd_gpu(const int n, const float_t* a, const float_t* b, float_t* out); // vector add
void axpy_gpu(const int n, const float_t a, const float_t* x, float_t* y);    // axpy
void relu_gpu(const int n, const float_t* in, float_t* out); // ReLU
void d_relu_gpu(const int n, const float_t* in_diff, const float_t* data, float_t* out_diff); // ReLU derivative
void leaky_relu_gpu(const int n, const float_t epsilon, const float_t* in, float_t* out); // Leaky ReLU
void d_leaky_relu_gpu(const int n, const float_t epsilon,
                      const float_t* in_diff, const float_t* data,
                      float_t* out_diff); // Leaky ReLU derivative
void dropout_gpu(int n, float scale, float drop_rate, const float* in, mask_t* masks, float* out); // dropout
void d_dropout_gpu(int n, float scale, const float* in, const mask_t* masks, float* out); // dropout derivative
void sgemm_gpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const float alpha,
               const float* A, const float* B, const float beta, float* C);
void csrmm_gpu(const int M, const int N, const int K, const int nnz,
               const float alpha, const float* A_nonzeros, const int* A_idx_ptr,
               const int* A_nonzero_idx, const float* B, const float beta,
               float* trans_C, float* C);
void softmax_cross_entropy_gpu(int len, int begin, int end,
                               const float_t* in_data, const mask_t* masks,
                               const label_t* labels, float_t* loss,
                               float_t* out_data);
void d_softmax_cross_entropy_gpu(int len, int bengin, int end,
                                 const mask_t* masks, const label_t* labels,
                                 const float_t* out_data, float_t* diff);
void sigmoid_cross_entropy_gpu(int len, int begin, int end,
                               const float_t* in_data, const mask_t* masks,
                               const label_t* labels, float_t* loss,
                               float_t* out_data);
void d_sigmoid_cross_entropy_gpu(int len, int bengin, int end,
                                 const mask_t* masks, const label_t* labels,
                                 const float_t* out_data, float_t* diff);
void scal_gpu(const int n, const float alpha, float* X);
void add_scalar_gpu(const int n, const float_t alpha, float_t* Y);
void rng_uniform_gpu(size_t n, const float_t a, const float_t b, float_t* r);
void l2_norm_gpu(size_t x, size_t y, const float_t* in, float_t* out);
void d_l2_norm_gpu(size_t x, size_t y, const float_t* in_data, float_t* in_diff, float_t* out_diff);
acc_t l2_norm_gpu(int n, const float_t* in);
acc_t masked_avg_loss_gpu(int begin, int end, int count, mask_t* masks, float_t* loss);

bool is_allocated_device(float_t* data);
void copy_masks_device(int n, mask_t* h_masks, mask_t*& d_masks);
void float_malloc_device(int n, float_t*& ptr);
void float_free_device(float_t*& ptr);
void copy_float_device(int n, float* h_ptr, float* d_ptr);
void copy_float_host(int n, const float* d_ptr, float* h_ptr);
void uint_malloc_device(int n, uint32_t*& ptr);
void uint_free_device(uint32_t*& ptr);
void copy_uint_device(int n, uint32_t* h_ptr, uint32_t* d_ptr);
void uint8_malloc_device(int n, uint8_t*& ptr);
void uint8_free_device(uint8_t*& ptr);
void copy_uint8_device(int n, uint8_t* h_ptr, uint8_t* d_ptr);
void gpu_rng_uniform(size_t n, float* r);
