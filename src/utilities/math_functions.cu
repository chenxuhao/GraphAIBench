#include <ctime>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <curand_kernel.h>
#include "util.h"
#include "cub/cub.cuh"
#include "gpu_context.h"
#include "math_functions.hh"
#include "gpu_device_functions.cuh"

__global__ void init_const_kernel(int n, float_t value, float_t* array) {
  CUDA_KERNEL_LOOP(i, n) { array[i] = value; }
}

void init_const_gpu(int n, float_t value, float_t* array) {
  init_const_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, value, array);
  CudaTest("solving init_const kernel failed");
}

__global__ void isnan_test(const int n, const float* data, bool* result) {
  CUDA_KERNEL_LOOP(i, n) {
    if (isnan(data[i]))
      *result = true;
  }
}

bool isnan_gpu(int n, const float_t* array) {
  bool *d_result, h_result = false;
  //cudaMalloc((void**)&d_result, sizeof(bool));
  malloc_device<bool>(1, d_result);
  cudaMemcpy(d_result, &h_result, sizeof(bool), cudaMemcpyHostToDevice);
  isnan_test<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, array, d_result);
  CudaTest("solving isnan_test kernel failed");
  cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
  return h_result;
}

void gpu_rng_uniform(size_t n, float_t* r) {
  CURAND_CHECK(curandGenerateUniform(gpu_context::curand_generator(), r, n));
}

void rng_uniform_gpu(size_t n, const float_t a, const float_t b, float_t* r) {
  CURAND_CHECK(curandGenerateUniform(gpu_context::curand_generator(), r, n));
  const float range = b - a;
  if (range != float_t(1))
    scal_gpu(n, range, r);
  if (a != float_t(0))
    add_scalar_gpu(n, a, r);
}

void gpu_rng_gaussian(const int n, const float_t mu, const float_t sigma, float_t* r) {
  CURAND_CHECK(curandGenerateNormal(gpu_context::curand_generator(), r, n, mu, sigma));
}

bool is_allocated_device(float_t* data) {
  if (data == NULL)
    return false;
  cudaPointerAttributes attributes;
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, data));
  if (attributes.devicePointer != NULL)
    return true;
  return false;
}

void float_malloc_device(int n, float*& ptr) {
  //std::cout << "allocating GPU memory: size = " << n << "\n";
  CUDA_CHECK(cudaMalloc((void**)&ptr, n * sizeof(float)));
}

void float_free_device(float_t*& ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = NULL; }

void copy_float_device(int n, float* h_ptr, float* d_ptr) {
  CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, n * sizeof(float_t), cudaMemcpyHostToDevice));
}

void copy_float_host(int n, const float* d_ptr, float* h_ptr) {
  CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
}

void uint_malloc_device(int n, uint32_t*& ptr) {
  CUDA_CHECK(cudaMalloc((void**)&ptr, n * sizeof(uint32_t)));
}

void uint_free_device(uint32_t*& ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = NULL; }

void copy_uint_device(int n, uint32_t* h_ptr, uint32_t* d_ptr) {
  CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void uint8_malloc_device(int n, uint8_t*& ptr) {
  CUDA_CHECK(cudaMalloc((void**)&ptr, n * sizeof(uint8_t)));
}

void uint8_free_device(uint8_t*& ptr) { CUDA_CHECK(cudaFree(ptr)); ptr = NULL; }

void copy_uint8_device(int n, uint8_t* h_ptr, uint8_t* d_ptr) {
  CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, n * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

void copy_masks_device(int n, mask_t* h_masks, mask_t*& d_masks) {
  assert(h_masks != NULL);
  CUDA_CHECK(cudaMalloc((void**)&d_masks, n * sizeof(mask_t)));
  CUDA_CHECK(cudaMemcpy(d_masks, h_masks, n * sizeof(mask_t), cudaMemcpyHostToDevice));
}

__global__ void setup_curand_kernel(const int n, curandState* state) {
  CUDA_KERNEL_LOOP(i, n) {
    // curand_init(1234, i, 0, &state[i]); // Each thread gets same seed 1234
    curand_init(7 + i, i, 0, &state[i]); // Each thread gets different seed
  }
}

__global__ void dropout_kernel(int n, float scale, float threshold,
                               float_t* rands, const float_t* in, mask_t* masks,
                               float_t* out) {
  CUDA_KERNEL_LOOP(i, n) {
    masks[i] = rands[i] > threshold ? 1 : 0;
    out[i]   = in[i] * masks[i] * scale;
  }
}

void dropout_gpu(int n, float scale, float dropout_rate, const float_t* in,
                 mask_t* masks, float_t* out) {
  float_t* rands;
  float_malloc_device(n, rands);
  gpu_rng_uniform(n, rands);
  dropout_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, scale, dropout_rate, rands, in, masks, out);
  CudaTest("solving dropout kernel failed");
  float_free_device(rands);
}

__global__ void d_dropout_kernel(int n, float scale, const float* in, 
                                 const mask_t* masks, float* out) {
  CUDA_KERNEL_LOOP(i, n) { out[i] = in[i] * masks[i] * scale; }
}

void d_dropout_gpu(int n, float scale, const float* in,
                   const mask_t* masks, float* out) {
  d_dropout_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, scale, in, masks, out);
  CudaTest("solving d_dropout kernel failed");
}

__global__ void l2_norm_kernel(const int n, const float* a, float* sum) {
  CUDA_KERNEL_LOOP(i, n) {
    float_t product = a[i] * a[i];
    atomicAdd(sum, product);
  }
}

acc_t l2_norm_gpu(int n, const float* x) {
  float sum = 0.0;
  CUBLAS_CHECK(cublasSnrm2(gpu_context::cublas_handle(), n, x, 1, &sum));
  return (acc_t)sum / 2.0;
}

__global__ void l2norm_kernel(int n, int dim, const float* in, float* out) {
  CUDA_KERNEL_LOOP(i, n) {
    float sum = 0;
    for (int j = 0; j < dim; j++)
      sum += in[i*dim+j] * in[i*dim+j];
    sum = sum < 1.0e-12 ? 1.0e-12 : sum;
    sum = sqrt(sum);
    assert(sum!=0.);
    for (int j = 0; j < dim; j++)
      out[i*dim+j] = in[i*dim+j]/sum;
  }
}

void l2norm(int n, int dim, const float* in, float* out) {
//  Timer t;
//  t.Start();
  l2norm_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, dim, in, out);
  CudaTest("solving l2norm kernel failed");
//  t.Stop();
//  time_ops[OP_NORM] += t.Seconds();
}

__global__ void d_l2norm_kernel(int n, int dim, const float* feat_in, const float* grad_in, float* grad_out) {
  CUDA_KERNEL_LOOP(i, n) {
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
}

void d_l2norm(int n, int dim, const float* feat_in, const float* grad_in, float* grad_out) {
//  Timer t;
//  t.Start();
  d_l2norm_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, dim, feat_in, grad_in, grad_out);
  CudaTest("solving d_l2norm kernel failed");
//  t.Stop();
//  time_ops[OP_NORM] += t.Seconds();
}

__global__ void bias_mv_kernel(int n, int len, float* x, float* b) {
  CUDA_KERNEL_LOOP(i, n) {
    for (int j = 0; j < len; j++)
      x[i*len+j] += b[j];
  }
}

void bias_mv(int n, int len, float* x, float* b) {
  Timer t;
  t.Start();
  // alternatively, cblas_dger
  bias_mv_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, len, x, b);
  CudaTest("solving bias_mv kernel failed");
  t.Stop();
  time_ops[OP_BIAS] += t.Seconds();
}

__global__ void reduce_sum_kernel(int n, int len, float* x, float* a) {
  CUDA_KERNEL_LOOP(i, n) {
    for (int j = 0; j < len; j++)
      a[j] += x[i*len+j];
  }
}

void reduce_sum(int n, int len, float* x, float* a) {
  Timer t;
  t.Start();
  init_const_gpu(len, 0., a);
  reduce_sum_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, len, x, a);
  CudaTest("solving reduce_sum kernel failed");
  t.Stop();
  time_ops[OP_REDUCE] += t.Seconds();
}

// flattern data into 1D before feed into the ReLU operater
__global__ void relu_kernel(const int n, const float_t* in, float_t* out) {
  CUDA_KERNEL_LOOP(i, n) { out[i] = in[i] > 0 ? in[i] : 0; }
}

void relu_gpu(const int n, const float_t* in, float_t* out) {
  Timer t;
  t.Start();
  relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, in, out);
  CudaTest("solving relu kernel failed");
  t.Stop();
  time_ops[OP_RELU] += t.Seconds();
}

__global__ void d_relu_kernel(const int n, const float_t* in_diff,
                              const float_t* data, float_t* out_diff) {
  CUDA_KERNEL_LOOP(i, n) { out_diff[i] = data[i] > 0 ? in_diff[i] : 0; }
}

void d_relu_gpu(const int n, const float_t* in_diff, const float_t* data,
                float_t* out_diff) {
  Timer t;
  t.Start();
  d_relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, in_diff, data,
                                                          out_diff);
  CudaTest("solving d_relu kernel failed");
  t.Stop();
  time_ops[OP_RELU] += t.Seconds();
}

// flattern data into 1D before feed into the ReLU operater
__global__ void leaky_relu_kernel(const int n, const float epsilon, const float* in, float* out) {
  CUDA_KERNEL_LOOP(i, n) { out[i] = in[i] > 0 ? in[i] : epsilon * in[i]; }
}

void leaky_relu_gpu(const int n, const float_t epsilon, const float_t* in,
                    float_t* out) {
  leaky_relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, epsilon, in,
                                                              out);
  CudaTest("solving leaky_relu kernel failed");
}

__global__ void d_leaky_relu_kernel(const int n, const float_t epsilon,
                                    const float_t* in_diff, const float_t* data,
                                    float_t* out_diff) {
  CUDA_KERNEL_LOOP(i, n) {
    out_diff[i] = in_diff[i] * (data[i] > 0 ? 1.0 : epsilon);
  }
}

void d_leaky_relu_gpu(const int n, const float_t epsilon,
                      const float_t* in_diff, const float_t* data,
                      float_t* out_diff) {
  d_leaky_relu_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      n, epsilon, in_diff, data, out_diff);
  CudaTest("solving d_leaky_relu kernel failed");
}

__global__ void matmul_kernel(int x, int y, int z, const float_t* A,
                              const float_t* B, float_t* C) {
  int row     = blockIdx.x * blockDim.x + threadIdx.x;
  int col     = blockIdx.y * blockDim.y + threadIdx.y;
  float_t sum = 0.0f;
  if (row < x && col < y) {
    for (int i = 0; i < z; i++) {
      sum += A[row * z + i] * B[i * y + col];
    }
  }
  C[row * y + col] = sum;
}

#define TILE_SZ 16
void matmul_naive(const size_t x, const size_t y, const size_t z,
                  const float* A, const float* B, float* C) {
  dim3 threadsPerBlock(TILE_SZ, TILE_SZ);
  dim3 blocksPerGrid((y - 1) / TILE_SZ + 1, (x - 1) / TILE_SZ + 1);
  matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(x, y, z, A, B, C);
  CudaTest("solving matmul kernel failed");
}

void sgemm_gpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const float alpha,
               const float* A, const float* B, const float beta, float* C) {
  Timer t;
  t.Start();
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(gpu_context::cublas_handle(), cuTransB, cuTransA,
                           N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
  CudaTest("solving sgemm kernel failed");
  t.Stop();
  time_ops[OP_DENSEMM] += t.Seconds();
}

void matmul(const size_t dim_x, const size_t dim_y, const size_t dim_z,
            const float* A, const float* B, float* C, bool transA, bool transB, bool accum) {
  const CBLAS_TRANSPOSE TransA = transA ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE TransB = transB ? CblasTrans : CblasNoTrans;
  sgemm_gpu(TransA, TransB, dim_x, dim_y, dim_z, 1.0, A, B, accum?1.0:0.0, C);
}

void csr2csc(int nrows, int ncols, int nnz,
             const float* values, const int* rowptr, const int* colidx,
             float* valuesT, int* rowptrT, int* colidxT) {
  size_t buffer_size;
  CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(gpu_context::cusparse_handle(), nrows, ncols, nnz, 
                 values, rowptr, colidx, valuesT, rowptrT, colidxT, CUDA_R_32F,
                 CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffer_size));
  void* buffer_temp = NULL;
  CUDA_CHECK(cudaMalloc(&buffer_temp, buffer_size));
  CUSPARSE_CHECK(cusparseCsr2cscEx2(gpu_context::cusparse_handle(), nrows, ncols, nnz, 
                 values, rowptr, colidx, valuesT, rowptrT, colidxT, CUDA_R_32F,
                 CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer_temp));
  CUDA_CHECK(cudaFree(buffer_temp));
}

// C = A x B, where A is a sparse matrix in CSR format, B is the dense matrix
// for vertex feature tensor. However, since cusparse only supports
// column-major, while feature tensor is stored in row-major, the actual
// computation is: C = trans(A x trans(B)). Currently, we use cublasSgeam to
// implement transposition and allocate intermediate workspace memory
// (transpose_C) for this.
void csrmm_gpu(const int M, const int N, const int K, const int nnz,
               const float alpha, const float* A_nonzeros, const int* A_idx_ptr,
               const int* A_nnz_idx, const float* B, const float beta,
               float* transpose_C, float* C, bool transA) {
  //std::cout << "[debug] csrmm_gpu m=" << M << ", n=" << N << ", k=" << K << ", nnz=" << nnz << "\n";

#if 0
  cusparseOperation_t TransA = transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t TransB = CUSPARSE_OPERATION_TRANSPOSE;
  CUSPARSE_CHECK(cusparseScsrmm2(gpu_context::cusparse_handle(), TransA, TransB,
                 M, N, K, nnz, &alpha, gpu_context::cusparse_matdescr(), 
                 A_nonzeros, A_idx_ptr, A_nnz_idx, B, N, &beta, transpose_C, M));
  // transpose C
  const float one  = 1.0;
  const float zero = 0.0;
  CUBLAS_CHECK(cublasSgeam(gpu_context::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_T,
                           N, M, &one, transpose_C, M, &zero, NULL, M, C, N));
//}
#else
//void csrmm_gpu_new(const int M, const int N, const int K, const int nnz, const float alpha,
//                   const float* A_nonzeros, const int* A_idx_ptr, const int* A_nnz_idx,
//                   const float* B, const float beta, float *transpose_C, float* C, bool transA) {
  //std::cout << "[debug]: csrmm_gpu_new m=" << M << ", n=" << N << ", k=" << K << ", nnz=" << nnz << "\n";
  cusparseOperation_t TransA = transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  //cusparseOperation_t TransB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t TransB = CUSPARSE_OPERATION_TRANSPOSE;
  cusparseSpMatDescr_t A_descr;
  cusparseDnMatDescr_t B_descr, C_descr;
  CUSPARSE_CHECK(cusparseCreateCsr(&A_descr, M, K, nnz, (void*)A_idx_ptr, (void*)A_nnz_idx, (void*)A_nonzeros,
                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  //CUSPARSE_CHECK(cusparseCreateDnMat(&B_descr, K, N, K, (void*)B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
  //CUSPARSE_CHECK(cusparseCreateDnMat(&C_descr, M, N, M, (void*)C, CUDA_R_32F, CUSPARSE_ORDER_ROW));
  CUSPARSE_CHECK(cusparseCreateDnMat(&B_descr, N, K, N, (void*)B, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CUSPARSE_CHECK(cusparseCreateDnMat(&C_descr, M, N, M, (void*)transpose_C, CUDA_R_32F, CUSPARSE_ORDER_COL));
  //size_t bufferSize = 0;
  //CUSPARSE_CHECK(cusparseSpMM_bufferSize(gpu_context::cusparse_handle(),
  //      TransA, TransB, (void*)&alpha, A_descr, B_descr, (void*)&beta,
  //      C_descr, CUDA_R_32F, CUSPARSE_COOMM_ALG1, &bufferSize));
  //cudaDeviceSynchronize();
  void* buffer = NULL;
  //if (bufferSize > 0) CUDA_CHECK(cudaMalloc(&buffer, bufferSize));
  CUSPARSE_CHECK(cusparseSpMM(gpu_context::cusparse_handle(), TransA, TransB,
                 (const void*)&alpha, A_descr, B_descr, (const void*)&beta,
                 //C_descr, CUDA_R_32F, CUSPARSE_COOMM_ALG1, buffer));
                 C_descr, CUDA_R_32F, CUSPARSE_CSRMM_ALG1, buffer));
  cudaDeviceSynchronize();
  const float one = 1.0;
  const float zero = 0.0;
  CUBLAS_CHECK(cublasSgeam(gpu_context::cublas_handle(), CUBLAS_OP_T,
               CUBLAS_OP_T, N, M, &one, transpose_C, M, &zero, NULL, M, C, N));
#endif
}

void spmm(size_t x, size_t y, size_t z, size_t nnz,
          float* A_nonzeros, int* A_idx_ptr, int* A_nnz_idx,
          const float* B, float* C, float* temp, bool transA, bool transB, bool accum) {
  //Timer t;
  //t.Start();
  //if (transA) csrmm_gpu_new(x, y, z, nnz, 1.0, A_nonzeros, A_idx_ptr, A_nnz_idx, B, accum?1.0:0.0, temp, C, transA);
  //else
    csrmm_gpu(x, y, z, nnz, 1.0, A_nonzeros, A_idx_ptr, A_nnz_idx, B, accum?1.0:0.0, temp, C, transA);
  //t.Stop();
  //time_ops[OP_SPARSEMM] += t.Seconds();
}

void gemv_gpu(const CBLAS_TRANSPOSE TransA, const int M, const int N,
              const float alpha, const float* A, const float* x,
              const float beta, float* y) {
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(gpu_context::cublas_handle(), cuTransA, N, M,
                           &alpha, A, N, x, 1, &beta, y, 1));
  CudaTest("solving gemv kernel failed");
}

void scal_gpu(const int N, const float alpha, float* X) {
  CUBLAS_CHECK(cublasSscal(gpu_context::cublas_handle(), N, &alpha, X, 1));
}

void dot_gpu(const int n, const float* x, const float* y, float* out) {
  CUBLAS_CHECK(cublasSdot(gpu_context::cublas_handle(), n, x, 1, y, 1, out));
}

void asum_gpu(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(gpu_context::cublas_handle(), n, x, 1, y));
}

void scale_gpu(const int n, const float alpha, const float* x, float* y) {
  CUBLAS_CHECK(cublasScopy(gpu_context::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(gpu_context::cublas_handle(), n, &alpha, y, 1));
}

__global__ void set_kernel(const int n, const float_t alpha, float_t* y) {
  CUDA_KERNEL_LOOP(i, n) { y[i] = alpha; }
}

__global__ void add_scalar_kernel(const int n, const float_t a, float_t* y) {
  CUDA_KERNEL_LOOP(i, n) { y[i] += a; }
}

void add_scalar_gpu(const int n, const float_t alpha, float_t* Y) {
  add_scalar_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, alpha, Y);
  CudaTest("solving add_scalar kernel failed");
}

__global__ void vadd_kernel(const int n, const float_t* a, const float_t* b,
                            float_t* y) {
  CUDA_KERNEL_LOOP(i, n) { y[i] = a[i] + b[i]; }
}

void vadd_gpu(const int n, const float_t* a, const float_t* b, float_t* y) {
  vadd_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, a, b, y);
  CudaTest("solving vadd kernel failed");
}

__global__ void axpy_kernel(const int n, const float_t a, const float_t* x,
                            float_t* y) {
  CUDA_KERNEL_LOOP(i, n) { y[i] = a * x[i] + y[i]; }
}

void axpy_gpu(const int n, const float_t a, const float_t* x, float_t* y) {
  // axpy_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, a, x, y);
  CUBLAS_CHECK(
      cublasSaxpy(gpu_context::cublas_handle(), n, &a, x, 1, y, 1));
  CudaTest("solving axpy kernel failed");
}

void copy_gpu(int len, const float_t* in, float_t* out) {
  CUDA_CHECK(
      cudaMemcpy(out, in, len * sizeof(float_t), cudaMemcpyDeviceToDevice));
}

// n: number of vectors
// len: length of vectors
// for each vector, do softmax to normalize the vector, and then compute a loss
__global__ void softmax_cross_entropy_kernel(int len, int begin, int end,
                                             const float* in, const mask_t* masks,
                                             const label_t* labels, float* loss, float* out) {
  CUDA_KERNEL_LOOP(i, end - begin) {
    int id = begin + i;
    if (masks == NULL || masks[id] == 1) { // masked normalize using softmax
      softmax_device(len, &in[len*id], &out[len*id]);
      cross_entropy_device(len, labels[id], &out[len*id], loss[id]);
    }
  }
}

__device__ float MaxOf2(float a, float b) {
  if(a > b) return a; else return b;
}

__global__ void softmax_cross_entropy_warp(int len, int begin, int end,
                                           const float* in, const mask_t* masks,
                                           const label_t* labels, float* loss, float* out) {
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const int warp_id = thread_id / WARP_SIZE;   // global warp index
  __shared__ float sdata[BLOCK_SIZE];
  int id = begin + warp_id;
  if (id < end && (masks == NULL || masks[id] == 1)) {
    float max = in[len*id];
    for (int i = 0; i < len; i += WARP_SIZE) {
      if (thread_lane+i < len) {
        max = in[len*id+thread_lane+i] > max ? in[len*id+thread_lane+i] : max;
      }
    }
    //if (id == begin) printf("max1=%f\n", max);
    max = MaxOf2(SHFL_DOWN(max, 16), max);
    max = MaxOf2(SHFL_DOWN(max, 8), max);
    max = MaxOf2(SHFL_DOWN(max, 4), max);
    max = MaxOf2(SHFL_DOWN(max, 2), max);
    max = MaxOf2(SHFL_DOWN(max, 1), max);
    max = SHFL(max, 0);
    //if (id == begin) printf("max=%f", max);
 
    float denominator = 0.0;
    for (int i = 0; i < len; i += WARP_SIZE) {
      if (thread_lane+i < len) {
        out[len*id+thread_lane+i] = expf(in[len*id+thread_lane+i] - max);
        denominator += out[len*id+thread_lane+i];
      }
    }
    // intra-warp reduction
    denominator += SHFL_DOWN(denominator, 16);
    denominator += SHFL_DOWN(denominator, 8);
    denominator += SHFL_DOWN(denominator, 4);
    denominator += SHFL_DOWN(denominator, 2);
    denominator += SHFL_DOWN(denominator, 1);
    denominator = SHFL(denominator, 0);
    //if (id == begin) printf("denominator=%f", denominator);
    for (int i = 0; i < len; i += WARP_SIZE) {
      if (thread_lane + i < len)
        out[len*id+thread_lane+i] /= denominator;
    }
    if (thread_lane == 0) {
      float p = out[len*id+labels[id]];
      loss[id] -= p == 0. ? logf(float(1e-10)) : logf(p);
    }
  }
}

void softmax_cross_entropy_gpu(int len, int begin, int end, const float_t* in,
                               const mask_t* masks, const label_t* labels,
                               float_t* loss, float_t* out) {
  //std::cout << "softmax_cross_entropy_gpu\n";
  //Timer t;
  //t.Start();
  //softmax_cross_entropy_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(len, begin, end, in, masks, labels, loss, out);
  softmax_cross_entropy_warp<<<(end-begin-1)/WARPS_PER_BLOCK+1, BLOCK_SIZE>>>(len, begin, end, in, masks, labels, loss, out);
  CudaTest("solving softmax_cross_entropy kernel failed");
  //t.Stop();
  //time_ops[OP_LOSS] += t.Seconds();
}

// n: number of vectors
// len: length of vectors
// for each vector, do sigmoid to normalize the vector, and then compute a loss using:
// https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
__global__ void sigmoid_cross_entropy_kernel(int len, int begin, int end,
                                             const float_t* in_data,
                                             const mask_t* masks,
                                             const label_t* labels,
                                             float_t* losses, float_t* out_data) {
  CUDA_KERNEL_LOOP(i, end - begin) {
    int id = begin + i;
    float loss = 0.0;
    if (masks == NULL || masks[id] == 1) { // masked
      auto idx = len * id;
      auto p = in_data + idx;
      auto y = labels + idx;
      sigmoid_device(len, p, out_data + idx);
      //cross_entropy_multi_device(len, labels, out_data + len * id, losses[id]);
      for (int j = 0; j < len; j++)
        loss -= p[j] * (float(y[j]) - (p[j] >= 0.)) - logf(1. + expf(p[j] - 2. * p[j] * (p[j] >= 0.)));
    }
    losses[id] = loss;
  }
}

void sigmoid_cross_entropy_gpu(int len, int begin, int end, const float_t* in,
                               const mask_t* masks, const label_t* labels,
                               float_t* loss, float_t* out) {
  sigmoid_cross_entropy_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(
                                 len, begin, end, in, masks, labels, loss, out);
  CudaTest("solving sigmoid_cross_entropy kernel failed");
}

__global__ void d_sigmoid_cross_entropy_kernel(int len, int begin, int end,
                                               const mask_t* masks,
                                               const label_t* labels,
                                               const float_t* preds,
                                               float_t* grad) {
  CUDA_KERNEL_LOOP(i, end - begin) {
    int id = begin + i;
    if (masks == NULL || masks[id] == 1) {   // masked
      auto idx = len * id;
      for (int j = 0; j < len; j++) {
        //auto pred = sigmoid_func(feat_in[idx+j]);
        auto pred = preds[idx+j];
        grad[idx+j] = (pred - float(labels[idx+j])) / (end - begin);
      }
    }
  }
}

void d_sigmoid_cross_entropy_gpu(int len, int begin, int end,
                                 const mask_t* masks, const label_t* labels,
                                 const float_t* out, float_t* diff) {
  //d_sigmoid_cross_entropy_warp<<<(end - begin - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(
  d_sigmoid_cross_entropy_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(
                                   len, begin, end, masks, labels, out, diff);
  CudaTest("solving d_sigmoid_cross_entropy kernel failed");
}

__global__ void d_cross_entropy_kernel(int len, int begin, int end,
                                       const mask_t* masks,
                                       const label_t* labels,
                                       const float_t* data, float_t* grad) {
  int base = begin * len;
  CUDA_KERNEL_LOOP(i, (end - begin) * len) {
    int id = begin + i / len;
    if (masks == NULL || masks[id] == 1) { // masked
      if (i % len == (int)labels[id])
        grad[i] = -1.0 / (data[i + base] + 1e-10);
      else
        grad[i] = 0.0;
      // d_cross_entropy_device(len, labels[id], data + len*id, grad + len*i);
    }
  }
}

__global__ void d_cross_entropy_warp(int len, int begin, int end,
                                     const mask_t* masks, const label_t* labels,
                                     const float_t* data, float_t* grad) {
  __shared__ float_t p[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];
  const int thread_id =
      BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);             // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  const int num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  for (int wid = warp_id; wid < end - begin; wid += num_warps) {
    int id   = begin + wid;
    int base = id * len;
    if (masks == NULL || masks[id] == 1) {
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len)
          p[warp_lane][pid] = data[base + pid];
      }
      __syncthreads();
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          if (pid == (int)labels[id])
            grad[wid * len + pid] = -1.0 / (p[warp_lane][pid] + 1e-10);
          else
            grad[wid * len + pid] = 0.0;
        }
      }
    }
  }
}

__global__ void d_softmax_kernel(int len, int begin, int end,
                                 const mask_t* masks, const float_t* data,
                                 const float_t* in_grad, float_t* out_grad) {
  CUDA_KERNEL_LOOP(i, end - begin) {
    int id = begin + i;
    if (masks == NULL || masks[id] == 1) { // masked
      d_softmax_device(len, data + len * id, in_grad + len * i,
                       out_grad + len * id);
    }
  }
}

__global__ void d_softmax_warp(int len, int begin, int end, const mask_t* masks,
                               const float_t* data, const float_t* in_grad,
                               float_t* out_grad) {
  __shared__ float_t p[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];
  __shared__ float_t d[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];
  const int thread_id =
      BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);             // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  const int num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  for (int wid = warp_id; wid < end - begin; wid += num_warps) {
    int id   = begin + wid;
    int base = id * len;
    if (masks == NULL || masks[id] == 1) {
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          p[warp_lane][pid] = data[base + pid];
          d[warp_lane][pid] = in_grad[wid * len + pid];
        }
      }
      __syncthreads();
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          float_t sum  = 0.0;
          float_t self = p[warp_lane][pid];
          for (int j = 0; j < len; j++) {
            float_t df =
                (j == pid) ? self * (1.0 - self) : -p[warp_lane][j] * self;
            sum += df * d[warp_lane][j];
          }
          out_grad[base + pid] = sum;
        }
      }
      __syncthreads();
    }
  }
}

// compute gradients (grad) based on predictions (pred) and graound truths (labels)
__global__ void d_softmax_cross_entropy_kernel(int len, int begin, int end,
                                               const mask_t* masks,
                                               const label_t* labels,
                                               const float* pred, float* grad) {
  CUDA_KERNEL_LOOP(i, end - begin) {
    int id = begin + i;
    if (masks == NULL || masks[id] == 1) {
      auto idx = len * i;
      for (int j = 0; j < len; j++)
        grad[idx+j] = (pred[idx+j] - (labels[i]==j?1.0:0.0)) / float(end - begin); // TODO: use count, not end-begin
    }
  }
}

__global__ void d_softmax_cross_entropy_warp(int len, int begin, int end,
                                             const mask_t* masks,
                                             const label_t* labels,
                                             const float_t* data,
                                             float_t* grad) {
  __shared__ float_t p[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];
  __shared__ float_t d[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];
  const int thread_id =
      BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);             // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  const int num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  for (int wid = warp_id; wid < end - begin; wid += num_warps) {
    int id   = begin + wid;
    int base = id * len;
    if (masks == NULL || masks[id] == 1) {
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len)
          p[warp_lane][pid] = data[base + pid];
      }
      __syncthreads();

      // cross entropy derivative
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          if (pid == (int)labels[id])
            d[warp_lane][pid] = -1.0 / (p[warp_lane][pid] + 1e-10);
          else
            d[warp_lane][pid] = 0.0;
        }
      }
      __syncthreads();

      // softmax derivative
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          float_t sum  = 0.0;
          float_t self = p[warp_lane][pid];
          for (int j = 0; j < len; j++) {
            float_t df =
                (j == pid) ? self * (1.0 - self) : -p[warp_lane][j] * self;
            sum += df * d[warp_lane][j];
          }
          grad[base + pid] = sum;
        }
      }
      __syncthreads();
    }
  }
}

void d_softmax_cross_entropy_gpu(int len, int begin, int end,
                                 const mask_t* masks, const label_t* labels,
                                 const float* out, float* diff) {
  //Timer t;
  //t.Start();
  //d_softmax_cross_entropy_warp<<<(end - begin - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(len, begin, end, masks, labels, out, diff);
  d_softmax_cross_entropy_kernel<<<CUDA_GET_BLOCKS(end-begin), BLOCK_SIZE>>>(len, begin, end, masks, labels, out, diff);
  CudaTest("solving d_softmax_cross_entropy kernel failed");
  //t.Stop();
  //time_ops[OP_LOSS] += t.Seconds();
}

__global__ void d_sigmoid_cross_entropy_warp(int len, int begin, int end,
                                             const mask_t* masks,
                                             const label_t* labels,
                                             const float_t* data,
                                             float_t* grad) {
  __shared__ float_t p[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];
  __shared__ float_t d[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];
  const int thread_id =
      BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);             // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  const int num_warps =
      (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  for (int wid = warp_id; wid < end - begin; wid += num_warps) {
    int id   = begin + wid;
    int base = id * len;
    if (masks == NULL || masks[id] == 1) {
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len)
          p[warp_lane][pid] = data[base + pid];
      }
      __syncthreads();

      // cross entropy derivative
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          // if (p[warp_lane][pid] == 0)
          d[warp_lane][pid] =
              -(float_t)labels[base + pid] / (p[warp_lane][pid] + 1e-10);
          // else d[warp_lane][pid] = -(float_t)labels[pid] / 1e-10;
        }
      }
      __syncthreads();

      // sigmoid derivative
      for (int i = 0; i < len; i += WARP_SIZE) {
        int pid = thread_lane + i;
        if (pid < len) {
          float_t self     = p[warp_lane][pid];
          float_t dp       = d[warp_lane][pid];
          grad[base + pid] = dp * self * (float_t(1) - self);
        }
      }
      __syncthreads();
    }
  }
}

__global__ void masked_avg_loss_kernel(int begin, int end, mask_t* masks, float* losses, acc_t* total) {
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  acc_t local_loss = 0.0;
  CUDA_KERNEL_LOOP(i, end - begin) {
    if (masks == NULL || masks[begin + i] == 1)
      local_loss += losses[begin + i];
  }
  acc_t block_loss = BlockReduce(temp_storage).Sum(local_loss);
  if (threadIdx.x == 0) atomicAdd(total, block_loss);
}

acc_t masked_avg_loss_gpu(int begin, int end, int count, mask_t* masks, float* losses) {
  assert(count > 0);
  acc_t h_total_loss = 0;
  acc_t *d_total_loss;
  //cudaMalloc((void**)&d_total_loss, sizeof(acc_t));
  malloc_device<acc_t>(1, d_total_loss);
  cudaMemcpy(d_total_loss, &h_total_loss, sizeof(acc_t), cudaMemcpyHostToDevice);
  masked_avg_loss_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(begin, end, masks, losses, d_total_loss);
  CudaTest("solving masked_avg_loss kernel failed");
  cudaMemcpy(&h_total_loss, d_total_loss, sizeof(acc_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  return h_total_loss / count;
}

__global__ void masked_accuracy_single_kernel(int num_classes, int begin, int end, mask_t* masks,
                                       float* preds, label_t* labels, acc_t* total) {
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  acc_t local_sum = 0.0;
  CUDA_KERNEL_LOOP(i, end - begin) {
    if (masks == NULL || masks[begin + i] == 1) {
      label_t pred = (label_t)argmax_device(num_classes, preds + (begin + i) * num_classes);
      if (pred == labels[begin + i]) local_sum += 1.0;
    }
  }
  acc_t block_sum = BlockReduce(temp_storage).Sum(local_sum);
  if (threadIdx.x == 0) atomicAdd(total, block_sum);
}

float masked_accuracy_single(int begin, int end, int count, int num_classes,
                             mask_t* masks, float* preds, label_t* labels) {
  assert(count > 0);
  acc_t h_total = 0;
  acc_t *d_total;
  //cudaMalloc((void**)&d_total, sizeof(acc_t));
  malloc_device<acc_t>(1, d_total);
  cudaMemcpy(d_total, &h_total, sizeof(acc_t), cudaMemcpyHostToDevice);
  masked_accuracy_single_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(
                                  num_classes, begin, end, masks, preds, labels, d_total);
  CudaTest("solving masked_accuracy_single kernel failed");
  cudaMemcpy(&h_total, d_total, sizeof(acc_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  return h_total / count;
}

typedef float f1count_t;
__global__ void masked_f1_score_kernel(int num_classes, int begin, int end, 
                                       mask_t* masks, float_t* preds, label_t* labels,
                                       f1count_t* true_positive, f1count_t* false_positive,
                                       f1count_t* false_negtive, f1count_t* true_negtive) {
  CUDA_KERNEL_LOOP(i, end - begin) {
    int id = begin + i;
    if (masks == NULL || masks[id] == 1) {
      for (size_t j = 0; j < num_classes; j++) {
        int idx = id * num_classes + j;
        if (labels[idx] == 1 && preds[idx] > 0.5) {
          atomicAdd(&true_positive[j], 1.0);
        } else if (labels[idx] == 0 && preds[idx] > 0.5) {
          atomicAdd(&false_positive[j], 1.0);
        } else if (labels[idx] == 1 && preds[idx] <= 0.5) {
          atomicAdd(&false_negtive[j], 1.0);
        } else {
          atomicAdd(&true_negtive[j], 1.0);
        }
      }
    }
  }
}

acc_t masked_f1_score_gpu(int num_classes, int begin, int end, int count,
                          mask_t* masks, float_t* preds, label_t* labels) {
  const float beta = 1.0;
  assert(count > 0);
  f1count_t* h_tp = new f1count_t[num_classes];
  f1count_t* h_fp = new f1count_t[num_classes];
  f1count_t* h_fn = new f1count_t[num_classes];
  f1count_t* h_tn = new f1count_t[num_classes];
  f1count_t *d_tp, *d_fp, *d_fn, *d_tn;
  float_malloc_device(num_classes, d_tp);
  float_malloc_device(num_classes, d_fp);
  float_malloc_device(num_classes, d_fn);
  float_malloc_device(num_classes, d_tn);
  init_const_gpu(num_classes, 0.0, d_tp);
  init_const_gpu(num_classes, 0.0, d_fp);
  init_const_gpu(num_classes, 0.0, d_fn);
  init_const_gpu(num_classes, 0.0, d_tn);
  masked_f1_score_kernel<<<CUDA_GET_BLOCKS(end - begin), CUDA_NUM_THREADS>>>(
      num_classes, begin, end, masks, preds, labels, d_tp, d_fp, d_fn, d_tn);
  CudaTest("solving masked_f1_score kernel failed");
  CUDA_CHECK(cudaMemcpy(h_tp, d_tp, num_classes * sizeof(f1count_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_fp, d_fp, num_classes * sizeof(f1count_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_fn, d_fn, num_classes * sizeof(f1count_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_tn, d_tn, num_classes * sizeof(f1count_t),
                        cudaMemcpyDeviceToHost));

  acc_t pNumerator     = 0.0;
  acc_t pDenominator   = 0.0;
  acc_t rNumerator     = 0.0;
  acc_t rDenominator   = 0.0;
  acc_t precisionMacro = 0.0;
  acc_t recallMacro    = 0.0;
  for (size_t i = 0; i < num_classes; i++) {
    acc_t fn = (acc_t)h_fn[i]; // false negtive
    acc_t fp = (acc_t)h_fp[i]; // false positive
    acc_t tp = (acc_t)h_tp[i]; // true positive
    //acc_t tn = (acc_t)h_tn[i]; // true positive

    precisionMacro += (tp+fp>0) ? (tp / (tp + fp)) : 0.;
    recallMacro    += (tp+fn>0) ? (tp / (tp + fn)) : 0.;
    pNumerator     += tp;
    pDenominator   += (tp + fp);
    rNumerator     += tp;
    rDenominator   += (tp + fn);
  }
  assert(num_classes>0);
  precisionMacro = precisionMacro / num_classes;
  recallMacro    = recallMacro / num_classes;
  acc_t f1_macro = (precisionMacro+recallMacro)>0 ? 
                   (((beta * beta) + 1) * precisionMacro * recallMacro) /
                   ((beta * beta) * precisionMacro + recallMacro) : 0.;
  acc_t recallMicro    = rDenominator>0 ? rNumerator / rDenominator : 0.;
  acc_t precisionMicro = pDenominator>0 ? pNumerator / pDenominator : 0.;
  acc_t f1_micro = (precisionMicro+recallMicro)>0 ?
                   (((beta * beta) + 1) * precisionMicro * recallMicro) /
                   ((beta * beta) * precisionMicro + recallMicro) : 0.;
  //std::cout << std::setprecision(3) << std::fixed << " (f1_micro: " << f1_micro
  //          << ", f1_macro: " << f1_macro << ") ";

  float_free_device(d_tp);
  float_free_device(d_fp);
  float_free_device(d_fn);
  float_free_device(d_tn);
  delete[] h_tp;
  delete[] h_fp;
  delete[] h_fn;
  delete[] h_tn;
  return f1_micro;
}

float masked_accuracy_multi(int begin, int end, int count, int num_classes,
                             mask_t* masks, float* preds, label_t* labels) {
  assert(count > 0);
  return masked_f1_score_gpu(num_classes, begin, end, count, masks, preds, labels);
} 

