#include "random.h"
#include "simd_functions.h"
#include "math_functions.h"

#define NOT_IMPLEMENTED                                                        \
  do {                                                                         \
    std::cout << "Not Implemented Yet";                                        \
    exit(1);                                                                   \
  } while (0);

void init_glorot(size_t dim_x, size_t dim_y, vec_t &weight, unsigned seed) {
  float init_range = sqrt(6.0 / (dim_x + dim_y));
#if 1
  std::default_random_engine rng(seed);
  std::uniform_real_distribution<float> dist(-init_range, init_range);
  for (size_t i = 0; i < dim_x; ++i)
    for (size_t j = 0; j < dim_y; ++j)
      weight[i * dim_y + j] = dist(rng);
#else
  int range_int=(int)(init_range*10000);
  srand(0);
  #pragma omp parallel for
  for (size_t i=0; i < dim_x; i++) {
    for (size_t j=0; j < dim_y; j++) {
      auto temp = float(rand()%(2*range_int) - range_int);
      weight[i * dim_y + j] = temp / 10000.;
    }
  }
#endif
}

int binary_search(int *colidx, int key, int begin, int end) {
  assert(begin < end);
  int l = begin;
  int r = end-1;
  while (r >= l) {
    int mid = l + (r - l) / 2;
    int value = colidx[mid];
    if (value == key) return mid;
    if (value < key) l = mid + 1;
    else r = mid - 1;
  }
  return -1;
}

void symmetric_csr_transpose(int N, int nnz, int* A_idx_ptr, int* A_nnz_idx,
                             float* A_nonzeros, float* &B_nonzeros) {
  B_nonzeros = new float[nnz];
#if 0//def USE_MKL
  sparse_matrix_t A;
  mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, N, N, 
      A_idx_ptr, A_idx_ptr+1, A_nnz_idx, A_nonzeros);
  sparse_matrix_t B;
  mkl_sparse_convert_csr(A, SPARSE_OPERATION_TRANSPOSE, &B);
  int * bi, *ei, *j;
  int row, col;
  sparse_index_base_t indextype;
  sparse_status_t status = mkl_sparse_s_export_csr(A, &indextype, &row, &col, &bi, &ei, &j, &B_nonzeros);
#else
  //std::vector<int> index(N+1);
  //std::copy(A_idx_ptr, A_idx_ptr+N+1, &index[0]);
  #pragma omp parallel for schedule(dynamic, 64)
  for (int src = 0; src < N; src++) {
    auto begin = A_idx_ptr[src];
    auto end = A_idx_ptr[src+1];
    for (auto e = begin; e != end; e++) {
      auto dst = A_nnz_idx[e];
      auto idx = binary_search(A_nnz_idx, src, A_idx_ptr[dst], A_idx_ptr[dst+1]);
      assert(idx != -1);
      B_nonzeros[idx] = A_nonzeros[e];
    }
  }
#endif
}

acc_t masked_f1_score(int begin, int end, int, int num_classes, 
                      mask_t* masks, float* pred, label_t* ground_truth);
 
float masked_accuracy_single(int begin, int end, int count, int num_classes,
                      mask_t* masks, float* preds, label_t* ground_truth) {
  float accuracy_all = 0.0;
  int num_samples = 0;
  #pragma omp parallel for reduction(+:accuracy_all,num_samples)
  for (int i=begin; i<end; i++) {
    if (masks == NULL || masks[i] == 1) {
      auto pred = argmax(num_classes, &preds[i*num_classes]);
      if (pred == ground_truth[i]) accuracy_all += 1.0;
      num_samples ++;
    }
  }
  return accuracy_all / float(num_samples);
}

float masked_accuracy_multi(int begin, int end, int count, int num_classes,
                            mask_t* masks, float* preds, label_t* labels) {
  return masked_f1_score(begin, end, count, num_classes, masks, preds, labels);
}
 
// dot product
float dot(int n, const float* x, const float* y) {
  float sum = 0;
  for (int i = 0; i < n; ++i) sum += x[i]*y[i];
  return sum;
/*
#ifdef __AVX512F__
  const int alignedN = n - n % 16;
  __m512 r16 = mul16<0>(x, y);
  for (int i = 0; i < alignedN; i += 16)
    r16 = fma16<0>(r16, &x[i], &y[i]);
  float sum = reduce16(r16);
  for (int i = alignedN; i < n; ++i) sum += x[i]*y[i];
  return sum;
#else
#ifdef __AVX2__
  const int alignedN = n - n % 8;
  __m256 r8 = mul8<0>(x, y);
  for (int i = 0; i < alignedN; i += 8)
    r8 = fma8<0>(r8, &x[i], &y[i]);
  float sum = reduce8(r8);
  for (int i = alignedN; i < n; ++i) sum += x[i]*y[i];
  return sum;
#else
  return cblas_sdot(n, x, 1, y, 1);
#endif
#endif
*/
}

int argmax(int num, const float* arr) {
  int max_idx = -1;
  t_data max = -INFINITY;
  for (int i = 0; i < num; i++) {
    if (arr[i] > max) {
      max = arr[i];
      max_idx = i;
    }
  }
  return max_idx;
}

//! wrapper function to call cblas_sgemm
void sgemm_cpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const float alpha,
               const float* A, const float* B, const float beta, float* C) {
  double t1 = omp_get_wtime();
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
  double t2 = omp_get_wtime();
  time_ops[OP_DENSEMM] += t2 - t1;
}

// A: x*z; B: z*y; C: x*y
void matmul_naive(const size_t x, const size_t y, const size_t z, 
                  const float* A, const float*B, float* C) {
  for (size_t i = 0; i < x; i++) {
    for (size_t j = 0; j < y; j++) {
      C[i * y + j] = 0;
      for (size_t k = 0; k < z; k++) {
        C[i * y + j] += A[i*z+k] * B[k*z+j];
      }
    }
  }
}

void matmul(const size_t dim_x, const size_t dim_y, const size_t dim_z,
            const float* A, const float* B, float* C, bool transA, bool transB, bool accum) {
  const CBLAS_TRANSPOSE TransA = transA ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE TransB = transB ? CblasTrans : CblasNoTrans;
  sgemm_cpu(TransA, TransB, dim_x, dim_y, dim_z, 1.0, A, B, accum?1.0:0.0, C);
}

void spmm(size_t x, size_t y, size_t z, size_t nnz,
          float* A_nonzeros, int* A_idx_ptr, int* A_nnz_idx,
          const float* B, float* C, float*temp, bool transA, bool transB, bool accum) {
  //double t1 = omp_get_wtime();
#ifdef USE_MKL
  // const char *matdescra = "GXXCX";//6 bytes
  // const char transa = 'N';
  // mkl_scsrmm(&transa, &x , &y, &z, &alpha, matdescra, A_nonzeros, A_nnz_idx, A_idx_ptr, A_idx_ptr+1, B, &y, &beta, C, &y);
  sparse_status_t status;
  bool is_row_major            = true;
  sparse_matrix_t csrA         = NULL;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  sparse_layout_t layout = (is_row_major ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR);
  status = mkl_sparse_s_create_csr(&csrA, indexing, x, z, A_idx_ptr, A_idx_ptr + 1, A_nnz_idx, A_nonzeros);
  if (status != SPARSE_STATUS_SUCCESS) {
    std::cout << "mkl_sparse_s_create_csr status :" << status << std::endl;
    exit(1);
  }
  sparse_operation_t transa = (transA ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE);
  struct matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  // descrA.mode = SPARSE_FILL_MODE_UPPER;
  // descrA.diag = SPARSE_DIAG_NON_UNIT;
  // mkl_sparse_set_mm_hint(csrA, transa, descrA, layout, y, 1);
  // mkl_sparse_optimize(csrA);
  status = mkl_sparse_s_mm(transa, 1.0, csrA, descrA, layout, B, y, y, 0.0, C, y);
  if (status != SPARSE_STATUS_SUCCESS) {
    std::cout << "mkl_sparse_s_create_csr status :" << status << std::endl;
    exit(1);
  }
  mkl_sparse_destroy(csrA);
#else
  #pragma omp parallel
  {
  vec_t neighbor(y);
  #pragma omp for schedule(dynamic, 64)
  for (size_t i = 0; i < x; i++) {
    clear_cpu(y, &C[i*y]);
    for (auto e = A_idx_ptr[i]; e != A_idx_ptr[i+1]; e++) {
      const auto j = A_nnz_idx[e];
      scale(y, A_nonzeros[e], &B[j*y], &neighbor[0]);
      vadd_cpu(y, &C[i*y], &neighbor[0], &C[i*y]);
    }
  }
  }
#endif
  //double t2 = omp_get_wtime();
  //time_ops[OP_SPARSEMM] += t2 - t1;
}

// matrix-vector multiply
void mvmul(const CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha,
           const float* A, const float* x, const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void bias_mv(int n, int len, float* x, float* b) {
  double t1 = omp_get_wtime();
  // alternatively, cblas_dger
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < len; j++) {
      x[i*len+j] += b[j];
    }
  }
  double t2 = omp_get_wtime();
  time_ops[OP_BIAS] += t2-t1;
}

#pragma omp declare reduction(vec_float_plus : std::vector<float> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
                              initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

void reduce_sum(int n, int len, float* x, vec_t &a) {
  double t1 = omp_get_wtime();
  #pragma omp parallel for
  for (int i = 0; i < len; i++)
    a[i] = 0.;
#if 1
  #pragma omp parallel for
  for (int i = 0; i < len; i++)
    for (int j = 0; j < n; j++)
      a[i] += x[j*len+i];
#else
  #pragma omp parallel for reduction(vec_float_plus : a)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < len; j++)
      a[j] += x[i*len+j];
#endif
  double t2 = omp_get_wtime();
  time_ops[OP_REDUCE] += t2-t1;
}

void vadd_cpu(int n, const float_t* a, const float_t* b, float_t* y) {
  //vsAdd(n, a, b, y);
#ifdef __AVX512F__
  const int alignedN = n - n % 16;
  for (int i = 0; i < alignedN; i += 16)
    _mm512_storeu_ps(&y[i], _mm512_add_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i])));
  for (int i = alignedN; i < n; ++i) y[i] = a[i] + b[i];
#else
#ifdef __AVX2__
  const int alignedN = n - n % 8;
  for (int i = 0; i < alignedN; i += 8)
    _mm256_storeu_ps(&y[i], _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
  for (int i = alignedN; i < n; ++i) y[i] = a[i] + b[i];
#else
  for (int i = 0; i < n; ++i) y[i] = a[i] + b[i];
#endif
#endif
}

template <typename T, typename U>
inline void atomic_add(T& x, U inc) {
  return __sync_fetch_and_add(&x, inc);
}

inline void atomic_float_add(float& x, float inc) {
  #pragma omp critical
  x = x + inc;
  /*
  volatile int64_t* ptr = reinterpret_cast<volatile int64_t*>(&x);
  int64_t expected = *ptr;
  while (true) {
    float val = binary_cast<float>(expected);
    int64_t new_val = binary_cast<int64_t>(val + inc);
    const int64_t actual = __sync_val_compare_and_swap(ptr, expected, new_val);
    if (actual == expected) return;
    expected = actual;
  }
  */
}

void atomic_vreduce_cpu(size_t n, const float* x, float* y) {
  for (size_t i = 0; i < n; ++i) {
    atomic_float_add(y[i], x[i]);
  }
}

void scal(size_t n, const float_t alpha, float_t* x) {
  cblas_sscal(n, alpha, x, 1);
}

void scaled_vadd_cpu(int n, const float a, const float* x, const float* y, float* z) {
#ifdef __AVX512F__
  const __m512 scalar = _mm512_set1_ps(a);
  const int alignedN = n - n % 16;
  for (int i = 0; i < alignedN; i += 16)
    _mm512_storeu_ps(&z[i], _mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(&x[i]), scalar), _mm512_loadu_ps(&y[i])));
  for (int i = alignedN; i < n; ++i) z[i] = a*x[i] + y[i];
#else
#ifdef __AVX2__
  const __m256 scalar = _mm256_set1_ps(a);
  const int alignedN = n - n % 8;
  for (int i = 0; i < alignedN; i += 8)
    _mm256_storeu_ps(&z[i], _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(&x[i]), scalar), _mm256_loadu_ps(&y[i])));
  for (int i = alignedN; i < n; ++i) z[i] = a*x[i] + y[i];
#else
  for (int i = 0; i < n; ++i) z[i] = a*x[i] + y[i];
#endif
#endif
}

void scale(int n, const float a, const float* x, float* y) {
#ifdef __AVX512F__
  const __m512 scalar = _mm512_set1_ps(a);
  const int alignedN = n - n % 16;
  for (int i = 0; i < alignedN; i += 16)
    _mm512_storeu_ps(&y[i], _mm512_mul_ps(_mm512_loadu_ps(&x[i]), scalar));
  for (int i = alignedN; i < n; ++i) y[i] = a * x[i];
#else
#ifdef __AVX2__
  const __m256 scalar = _mm256_set1_ps(a);
  const int alignedN = n - n % 8;
  for (int i = 0; i < alignedN; i += 8)
    _mm256_storeu_ps(&y[i], _mm256_mul_ps(_mm256_loadu_ps(&x[i]), scalar));
  for (int i = alignedN; i < n; ++i) y[i] = a * x[i];
#else
  for (int i = 0; i < n; i++) y[i] = a * x[i];
#endif
#endif
  //cblas_scopy(n, x, 1, y, 1);
  //cblas_sscal(n, a, y, 1);
}

void axpy(size_t n, const float_t a, float_t* x, float_t* y) {
  cblas_saxpy(n, a, x, 1, y, 1);
}

void copy_cpu(size_t n, const float_t* in, float_t* out) {
  // std::copy(in, in + n, out);
  // memcpy(out, in, sizeof(float_t) * n);
  cblas_scopy(n, in, 1, out, 1);
}

void clear_cpu(int n, float* in) {
#ifdef __AVX512F__
  const __m512 scalar = _mm512_set1_ps(0);
  const int alignedN = n - n % 16;
  for (int i = 0; i < alignedN; i += 16)
    _mm512_storeu_ps(&in[i], scalar);
  for (int i = alignedN; i < n; ++i) in[i] = 0;
#else
#ifdef __AVX2__
  const __m256 scalar = _mm256_set1_ps(0);
  const int alignedN = n - n % 8;
  for (int i = 0; i < alignedN; i += 8)
    _mm256_storeu_ps(&in[i], scalar);
  for (int i = alignedN; i < n; ++i) in[i] = 0;
#else
  for (int i = 0; i < n; i++) in[i] = 0;
#endif
#endif
  //std::fill(in, in + n, 0);
  // memset(in, 0, n*sizeof(float_t));
}

static PerThreadRNG* per_thread_rng = nullptr;
inline uint8_t bernoulli(float_t p) {
  if (!per_thread_rng) {
    per_thread_rng = new PerThreadRNG();
  }
  return per_thread_rng->get_number() > p ? 1 : 0;
}

inline void rng_bernoulli(const int n, float p, mask_t *r) {
  boost::bernoulli_distribution<float> random_distribution(p);
  boost::variate_generator<rng_t*, boost::bernoulli_distribution<float> >
      variate_generator(rng(), random_distribution);
  for (int i = 0; i < n; ++i) r[i] = static_cast<mask_t>(variate_generator());
}

void dropout(size_t m, float scale, float dropout_rate, const float* in,
             mask_t* masks, float* out) {
  //for (size_t i = 0; i < m; ++i) masks[i] = bernoulli(dropout_rate);
  rng_bernoulli(m, 1-dropout_rate, masks);
  for (size_t i = 0; i < m; ++i)
    out[i] = in[i] * (float_t)masks[i] * scale;
}

void dropout_cpu(size_t n, size_t m, float scale, float dropout_rate,
                 const float* in, mask_t* masks, float* out) {
  double t1 = omp_get_wtime();
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    rng_bernoulli(m, 1-dropout_rate, &masks[i*m]);
  }
  #pragma omp parallel for
  for (size_t i = 0; i < n*m; i++)
    out[i] = in[i] * (float_t)masks[i] * scale;
  double t2 = omp_get_wtime();
  time_ops[OP_DROPOUT] += t2 - t1;
}

void d_dropout(size_t m, float scale, const float* in, const mask_t* masks, float* out) {
  for (size_t i = 0; i < m; ++i)
    out[i] = in[i] * (float_t)masks[i] * scale;
}

void d_dropout_cpu(size_t n, size_t m, float scale, const float* in,
                   const mask_t* masks, float* out) {
  double t1 = omp_get_wtime();
  #pragma omp parallel for
  for (size_t i = 0; i < n*m; i++)
    out[i] = in[i] * (float_t)masks[i] * scale;
  double t2 = omp_get_wtime();
  time_ops[OP_DROPOUT] += t2 - t1;
}

void relu_cpu(size_t n, const float* in, float* out) {
  double t1 = omp_get_wtime();
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    // TODO: vectorize
    out[i] = std::max(in[i], float(0));
  }
  double t2 = omp_get_wtime();
  time_ops[OP_RELU] += t2 - t1;
}

void d_relu_cpu(size_t n, const float* in, const float* data, float* out) {
  double t1 = omp_get_wtime();
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    // TODO: vectorize
    // check if original data greater than 0; if so keep grad
    out[i] = data[i] > float_t(0) ? in[i] : float(0);
  }
  double t2 = omp_get_wtime();
  time_ops[OP_RELU] += t2 - t1;
}

void leaky_relu(float_t epsilon, float_t in, float_t &out) {
  out = in > 0.0 ? in : epsilon * in;
}

void d_leaky_relu(float_t epsilon, float_t in, float_t data, float_t &out) {
  out = in * (data > 0.0 ? 1.0 : epsilon);
}

void leaky_relu_cpu(size_t n, float epsilon, const float* in, float* out) {
  // TODO: vectorize
  for (size_t i = 0; i < n; i++)
    out[i] = in[i] > 0 ? in[i] : epsilon * in[i];
}

void d_leaky_relu_cpu(size_t n, float epsilon, const float* in, const float* data, float* out) {
  // TODO: vectorize
  for (size_t i = 0; i < n; i++)
    out[i] = in[i] * (data[i] > float(0) ? float(1) : epsilon);
}

void softmax(size_t n, const float* input, float* output) {
  const float max = *std::max_element(input, input + n);
  float denominator(0);
  for (size_t i = 0; i < n; i++) {
    output[i] = expf(input[i] - max);
    denominator += output[i];
  }
  for (size_t i = 0; i < n; i++)
    output[i] /= denominator;
}

void d_softmax(int n, const float* p, const float* dp, float* dy) {
#ifdef __AVX512F__
  float score_sum = 0.;
  for (int i = 0; i < n; i++)
    score_sum += p[i]*dp[i];
  for (int i = 0; i < n; i++) {
    float x = p[i] * (1.0 - p[i]) * dp[i];
    dy[i] = x - (score_sum - p[i]*dp[i]) * p[i];
  }
#else 
  vec_t df(n, 0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      df[j] = (j == i) ? p[i] * (float(1) - p[i]) : -p[j] * p[i];
    }
    dy[i] = dot(n, dp, &df[0]);
  }
#endif
}

// Sigmoid
void sigmoid(size_t n, const float* in, float* out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = 1. / (1. + expf(-in[i]));
  }
}

void d_sigmoid(size_t n, const float*, const float* p, float* dy, const float* dp) {
  for (size_t i = 0; i < n; i++) {
    dy[i] = dp[i] * p[i] * (float(1) - p[i]);
  }
}

// cross-entropy loss function for multi-class classification
// y: ground truth
// p: predicted probability
float cross_entropy(size_t n, const float* y, const float* p) {
  float loss = 0.0;
  for (size_t i = 0; i < n; i++) {
    if (y[i] == float(0))
      continue;
    if (p[i] == float(0))
      loss -= y[i] * logf(float(1e-10));
    else
      loss -= y[i] * logf(p[i]);
  }
  return loss;
}

void d_cross_entropy(size_t n, const float* y, const float* p, float* d) {
  for (size_t i = 0; i < n; i++) {
    d[i] = -y[i] / (p[i] + float(1e-10));
  }
}

// Computes sigmoid cross entropy 
// https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
float sigmoid_cross_entropy(size_t n, const label_t* y, const float* p) {
  float loss = 0.0;
  for (size_t i = 0; i < n; i++) {
    loss -= p[i] * (float(y[i]) - (p[i] >= 0.)) - logf(1. + expf(p[i] - 2. * p[i] * (p[i] >= 0.)));
  }
  return loss;
}

// dummy functions
void float_malloc_device(int, float*&) {}
void copy_float_device(int, float_t*, float_t*) {}
void uint8_malloc_device(int, uint8_t*&) {}
void copy_uint8_device(int, mask_t*, mask_t*) {}

// Compute the F1 score, also known as balanced F-score or F-measure
// The F1 score can be interpreted as a weighted average of the precision and
// recall, where an F1 score reaches its best value at 1 and worst score at 0.
// The relative contribution of precision and recall to the F1 score are equal.
// The formula for the F1 score is:
// F1 = 2 * (precision * recall) / (precision + recall)
// where precision = TP / (TP + FP), recall = TP / (TP + FN)
// TP: true positive; FP: false positive; FN: false negative.
// In the multi-class and multi-label case, this is the weighted average of the
// F1 score of each class. Please refer to
// https://sebastianraschka.com/faq/docs/multiclass-metric.html,
// http://pageperso.lif.univ-mrs.fr/~francois.denis/IAAM1/scikit-learn-docs.pdf (p.1672)
// and https://github.com/ashokpant/accuracy-evaluation-cpp/blob/master/src/evaluation.hpp
acc_t masked_f1_score(int begin, int end, int, int num_classes,
                      mask_t* masks, float* pred, label_t* ground_truth) {
  // TODO dist version; make aware of distributed execution
  double precision_cls(0.), recall_cls(0.), f1_accum(0.);
  int tp_accum(0), fn_accum(0), fp_accum(0), tn_accum(0);
  for (int col = 0; col < num_classes; col++) {
    int tp_cls(0), fp_cls(0), fn_cls(0), tn_cls(0);
    for (auto row = begin; row < end; row++) {
      if (masks == NULL || masks[row] == 1) {
        auto idx = row * num_classes + col;
        if (ground_truth[idx] == 1 && pred[idx] > 0.5) {
          //__sync_fetch_and_add(&tp_cls, 1);
          tp_cls += 1;
        } else if (ground_truth[idx] == 0 && pred[idx] > 0.5) {
          //__sync_fetch_and_add(&fp_cls, 1);
          fp_cls += 1;
        } else if (ground_truth[idx] == 1 && pred[idx] <= 0.5) {
          //__sync_fetch_and_add(&fn_cls, 1);
          fn_cls += 1;
        } else if (ground_truth[idx] == 0 && pred[idx] <= 0.5) {
          //__sync_fetch_and_add(&tn_cls, 1);
          tn_cls += 1;
        }
      }
    }
    tp_accum += tp_cls;
    fn_accum += fn_cls;
    fp_accum += fp_cls;
    tn_accum += tn_cls;
    precision_cls = tp_cls + fp_cls > 0 ? (double)tp_cls / (double)(tp_cls + fp_cls) : 0.;
    recall_cls = tp_cls + fn_cls > 0 ? (double)tp_cls / (double)(tp_cls + fn_cls) : 0.;
    f1_accum += recall_cls + precision_cls > 0. ?
                2. * (recall_cls * precision_cls) / (recall_cls + precision_cls) : 0.;
  }
  double f1_macro = f1_accum / (double)num_classes;
  // double accuracy_mic = (double)(tp_accum+tn_accum)/(double)(tp_accum+tn_accum+fp_accum+fn_accum);
  double precision_mic = tp_accum + fp_accum > 0 ? (double)tp_accum / (double)(tp_accum + fp_accum) : 0.;
  double recall_mic = tp_accum + fn_accum > 0 ? (double)tp_accum / (double)(tp_accum + fn_accum) : 0.;
  double f1_micro = recall_mic + precision_mic > 0.
          ? 2. * (recall_mic * precision_mic) / (recall_mic + precision_mic) : 0.;
  //std::cout << std::setprecision(3) << std::fixed
  //          << " (f1_micro:" << f1_micro << ", f1_macro: " << f1_macro << ")\n";
  return f1_micro;
}

