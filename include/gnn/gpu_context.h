#pragma once

#include "cutils.h"
class gpu_context {
 public:
  gpu_context();
  ~gpu_context() {}
  static cublasHandle_t cublas_handle_;     // used to call cuBLAS
  static cusparseHandle_t cusparse_handle_; // used to call cuSPARSE
  static cusparseMatDescr_t cusparse_matdescr_; // used to call cuSPARSE
  static curandGenerator_t curand_generator_; // used to generate random numbers on GPU
  inline static cublasHandle_t cublas_handle() { return cublas_handle_; }
  inline static cusparseHandle_t cusparse_handle() { return cusparse_handle_; }
  inline static cusparseMatDescr_t cusparse_matdescr() { return cusparse_matdescr_; }
  inline static curandGenerator_t curand_generator() { return curand_generator_; }
};

