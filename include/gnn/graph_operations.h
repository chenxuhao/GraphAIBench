#pragma once
#include "lgraph.h"
#include "cutils.h"
#include "gpu_device_functions.cuh"
#define THRESHOLD 8
// if the vertex degree > BLOCK_SIZE, use the entire CTA to process it
template <typename DType>
__device__ __forceinline__ void reduce_cta(int n, int len, Graph g, const DType* scores, const DType* in, DType* out) {
//__global__ void reduce_cta(int n, int len, Graph g, const DType* scores, const DType* in, DType* out) {
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  __shared__ int owner;
  __shared__ int sh_vertex;
  __shared__ DType sum[WARPS_PER_BLOCK][MAX_NUM_CLASSES];
  //__shared__ DType block_sum[MAX_NUM_CLASSES];
  owner = -1;
  int size = 0;
  int v = warp_id; // every warp takes a vertex v
  if (v < n) size = g.getDegree(v);
  while (true) {
    if (size > THRESHOLD)
    //if (size > 0)
      owner = warp_lane;
    __syncthreads();
    if (owner == -1) break;
    __syncthreads();
    if (owner == warp_lane) {
      sh_vertex = v;
      owner = -1;
      size = 0;
    }
    for (int i = 0; i < len; i+=WARP_SIZE) if (thread_lane+i < len) sum[warp_lane][thread_lane+i] = 0.;
    __syncthreads();
    int begin = g.edge_begin(sh_vertex);
    int end = g.edge_end(sh_vertex);
    int degree = end - begin;
    // each warp processes an edge at a time
    for (int i = 0; i < degree; i += WARPS_PER_BLOCK) {
      if (i+warp_lane < degree) {
        int e = begin + i + warp_lane;
        int dst = g.getEdgeDst(e);
        DType score = scores[e];
        for (int j = 0; j < len; j += WARP_SIZE) {
          if (thread_lane+j < len)
            sum[warp_lane][thread_lane+j] += in[dst*len+thread_lane+j] * score;
        }
      }
    }
    // inter-warp reduction
    //for (int i = 0; i < len; i+=BLOCK_SIZE) if (threadIdx.x+i < len) block_sum[threadIdx.x+i] = 0.; 
    __syncthreads();
    if (warp_lane == 0) {
      for (int i = 1; i < WARPS_PER_BLOCK; i ++) {
        for (int j = 0; j < len; j += WARP_SIZE) {
          if (thread_lane+j < len)
            //block_sum[thread_lane+j] += sum[i][thread_lane+j];
            sum[0][thread_lane+j] += sum[i][thread_lane+j];
        }
      }
    }
    __syncthreads();
    for (int i = 0; i < len; i += BLOCK_SIZE)
      //if (threadIdx.x+i < len) out[sh_vertex*len+threadIdx.x+i] = block_sum[threadIdx.x+i];
      if (threadIdx.x+i < len) out[sh_vertex*len+threadIdx.x+i] = sum[0][threadIdx.x+i];
  }
}

__inline__ __device__ int binary_search_device(Graph graph, index_t key, index_t begin, index_t end) {
  assert(begin < end);
  int l = begin;
  int r = end-1;
  while (r >= l) {
    int mid = l + (r - l) / 2;
    index_t value = graph.getEdgeDst(mid);
    if (value == key) return mid;
    if (value < key) l = mid + 1;
    else r = mid - 1;
  }
  return -1;
}

template <typename DType>
__global__ void update_all_gcn(size_t n, int len, Graph g, const DType* in, DType* out) {
  __shared__ index_t ptrs[BLOCK_SIZE / WARP_SIZE][2];
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  const int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  for (int src = warp_id; src < n; src += num_warps) {
    //DType a = g.get_vertex_data(src);
    if (thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(src + thread_lane);
    __syncthreads();
    const index_t row_begin = ptrs[warp_lane][0];
    const index_t row_end   = ptrs[warp_lane][1];
    index_t base_src        = src * len;
    for (index_t offset = row_begin; offset < row_end; offset++) {
      index_t dst = g.getEdgeDst(offset);
      //DType b = a * DType(g.get_vertex_data(dst));
      DType b = g.get_edge_data(dst);
      index_t base_dst = dst * len;
      for (int i = 0; i < len; i += WARP_SIZE)
        if (thread_lane + i < len)
          out[base_src + thread_lane + i] += in[base_dst + thread_lane + i] * b;
    }
  }
}

template <typename DType>
__global__ void update_all_sage(size_t n, int len, Graph g, const DType* in, DType* out, bool trans) {
  __shared__ index_t ptrs[BLOCK_SIZE / WARP_SIZE][2];
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  const int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  for (int src = warp_id; src < n; src += num_warps) {
    DType c = 0.0;
    if (!trans) c = 1.0 / DType(g.getDegree(src));
    if (thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(src + thread_lane);
    __syncthreads();
    const index_t row_begin = ptrs[warp_lane][0];
    const index_t row_end   = ptrs[warp_lane][1];
    index_t base_src        = src * len;
    for (index_t offset = row_begin; offset < row_end; offset++) {
      index_t dst = g.getEdgeDst(offset);
      if (trans) c = 1.0 / DType(g.getDegree(dst));
      index_t base_dst = dst * len;
      for (int i = 0; i < len; i += WARP_SIZE)
        if (thread_lane + i < len)
          out[base_src + thread_lane + i] += in[base_dst + thread_lane + i] * c;
    }
  }
}

template <typename DType>
__global__ void reduce_thread(size_t n, int len, Graph g, const DType* scores, const DType* in, DType* out) {
  CUDA_KERNEL_LOOP(src, n) {
    auto begin = g.edge_begin(src);
    auto end   = g.edge_end(src);
    for (auto e = begin; e != end; e++) {
      auto dst = g.getEdgeDst(e);
      for (int i = 0; i < len; i++)
        out[src*len+i] += scores[e] * in[dst*len+i];
    }
  }
}

template <typename DType>
__global__ void reduce_warp(size_t n, int len, Graph g, const DType* scores, const DType* in, DType* out) {
  reduce_cta<DType>(n, len, g, scores, in, out);
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  //const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  //const int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps
  int src = warp_id;
  if (src < n) {
  //for (int src = warp_id; src < n; src += num_warps) {
    auto begin = g.edge_begin(src);
    auto end = g.edge_end(src);
    if (end - begin <= THRESHOLD) {
      for (auto e = begin; e < end; e++) {
        auto dst = g.getEdgeDst(e);
        for (int i = 0; i < len; i += WARP_SIZE) {
          if (thread_lane+i < len)
            out[src*len+thread_lane+i] += in[dst*len+thread_lane+i] * scores[e];
        }
      }
    }
  }
}

template <typename DType>
__global__ void compute_scores_grad_naive(size_t n, int len, Graph g, const DType* feat, const DType* grad, DType* scores_grad) {
  CUDA_KERNEL_LOOP(src, n) {
    for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
      auto dst = g.getEdgeDst(e);
      scores_grad[e] = dot_product(len, &grad[src*len], &feat[dst*len]);
    }
  }
}

template <typename DType>
__global__ void compute_scores_grad_warp(size_t n, int len, Graph g, const DType* feat, const DType* grad, DType* out) {
  __shared__ index_t ptrs[BLOCK_SIZE / WARP_SIZE][2];
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  const int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  for (int src = warp_id; src < n; src += num_warps) {
    if (thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(src + thread_lane);
    __syncthreads();
    auto row_begin = ptrs[warp_lane][0];
    auto row_end   = ptrs[warp_lane][1];
    index_t base_src = src * len;
    for (auto e = row_begin; e < row_end; e++) {
      auto dst = g.getEdgeDst(e);
      DType score = 0.;
      for (int i = 0; i < len; i += WARP_SIZE) {
        if (thread_lane+i < len)
          score += feat[dst*len + thread_lane+i] * grad[base_src + thread_lane+i];
      }
      // intra-warp reduction
      score += SHFL_DOWN(score, 16);
      score += SHFL_DOWN(score, 8);
      score += SHFL_DOWN(score, 4);
      score += SHFL_DOWN(score, 2);
      score += SHFL_DOWN(score, 1);
      score = SHFL(score, 0);
      if (thread_lane == 0) out[e] = score;
    }
  }
}

template <typename DType>
__global__ void compute_attn_score_naive(size_t n, int len, Graph g, DType eps, DType scale,
                                         DType attn_drop, const DType* alpha_l, const DType* alpha_r, 
                                         DType* scores, DType* temp_scores, DType* norm_scores, 
                                         DType* rands, mask_t *attn_masks, const DType* in) {
  CUDA_KERNEL_LOOP(src, n) {
    auto begin = g.edge_begin(src);
    auto end = g.edge_end(src);
    DType src_score = 0.;
    for (int i = 0; i < len; i++) src_score += alpha_l[i] * in[src*len+i];
    for (auto e = begin; e != end; e++) {
      auto dst = g.getEdgeDst(e);
      DType dst_score = 0.;
      for (int i = 0; i < len; i++) dst_score += alpha_r[i] * in[dst*len+i];
      temp_scores[e] = src_score + dst_score;
      scores[e] = temp_scores[e] > 0 ? temp_scores[e] : eps * temp_scores[e];
    }
    auto deg_src = g.getDegree(src);
    softmax((int)deg_src, &scores[begin], &norm_scores[begin]);
    if (attn_drop > 0.)
      dropout(deg_src, scale, attn_drop, &rands[begin], &norm_scores[begin], &attn_masks[begin], &norm_scores[begin]);
  }
}

template <typename DType>
__global__ void compute_attn_score_warp(size_t n, int len, Graph g, DType eps, DType scale,
                                        DType attn_drop, const DType* alpha_l, const DType* alpha_r, 
                                        DType* scores, DType* temp_scores, DType* norm_scores, 
                                        DType* rands, mask_t *attn_masks, const DType* in) {
  __shared__ index_t ptrs[BLOCK_SIZE / WARP_SIZE][2];
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  const int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  for (int src = warp_id; src < n; src += num_warps) {
    if (thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(src + thread_lane);
    __syncthreads();
    auto begin = ptrs[warp_lane][0];
    auto end   = ptrs[warp_lane][1];
    DType src_score = 0.;
    for (int i = 0; i < len; i += WARP_SIZE) {
      if (thread_lane+i < len)
        src_score += alpha_l[thread_lane+i] * in[src*len+thread_lane+i];
    }
    // intra-warp reduction
    src_score += SHFL_DOWN(src_score, 16);
    src_score += SHFL_DOWN(src_score, 8);
    src_score += SHFL_DOWN(src_score, 4);
    src_score += SHFL_DOWN(src_score, 2);
    src_score += SHFL_DOWN(src_score, 1);
    src_score = SHFL(src_score, 0);

    DType max = -1.;
    for (auto e = begin; e < end; e++) {
      auto dst = g.getEdgeDst(e);
      DType dst_score = 0.;
      for (int i = 0; i < len; i += WARP_SIZE) {
        if (thread_lane+i < len)
          dst_score += alpha_r[thread_lane+i] * in[dst*len+thread_lane+i];
      }
      // intra-warp reduction
      dst_score += SHFL_DOWN(dst_score, 16);
      dst_score += SHFL_DOWN(dst_score, 8);
      dst_score += SHFL_DOWN(dst_score, 4);
      dst_score += SHFL_DOWN(dst_score, 2);
      dst_score += SHFL_DOWN(dst_score, 1);
      dst_score = SHFL(dst_score, 0);
 
      DType temp_score = src_score + dst_score;
      if (thread_lane == 0) temp_scores[e] = temp_score;
      temp_score = temp_score > 0 ? temp_score : eps * temp_score;
      if (thread_lane == 0) scores[e] = temp_score;
      if (temp_score > max) max = temp_score;
    }

    // Softmax
    auto deg_src = g.getDegree(src);
    DType denominator = 0.0;
    for (int i = 0; i < deg_src; i += WARP_SIZE) {
      if (thread_lane + i < deg_src) {
        norm_scores[begin+thread_lane+i] = expf(scores[begin+thread_lane+i] - max);
        denominator += norm_scores[begin+thread_lane+i];
      }
    }
    // intra-warp reduction
    denominator += SHFL_DOWN(denominator, 16);
    denominator += SHFL_DOWN(denominator, 8);
    denominator += SHFL_DOWN(denominator, 4);
    denominator += SHFL_DOWN(denominator, 2);
    denominator += SHFL_DOWN(denominator, 1);
    denominator = SHFL(denominator, 0);
    for (int i = 0; i < deg_src; i += WARP_SIZE) {
      if (thread_lane + i < deg_src) {
        norm_scores[begin+thread_lane+i] /= denominator;
      }
    }

    // Dropout 
    if (attn_drop > 0.) {
      for (int i = 0; i < deg_src; i+= WARP_SIZE) {
        if (thread_lane + i < deg_src) {
          mask_t mask = rands[begin+thread_lane+i] > attn_drop ? 1 : 0;
          attn_masks[begin+thread_lane+i] = mask;
          auto score = norm_scores[begin+thread_lane+i];
          norm_scores[begin+thread_lane+i] = score * mask * scale;
        }
      }
    }
  }
}

template <typename DType>
__global__ void symmetric_csr_transpose_kernel(size_t n, Graph g, const DType* scores, DType* trans_scores) {
  CUDA_KERNEL_LOOP(src, n) {
    auto begin = g.edge_begin(src);
    auto end = g.edge_end(src);
    for (auto e = begin; e != end; e++) {
      auto dst = g.getEdgeDst(e);
      auto l = g.edge_begin(dst);
      auto r = g.edge_end(dst);
      //auto ei = binary_search_device(g, src, l, r);
      r = r-1;
      int mid;
      while (r >= l) {
        mid = l + (r - l) / 2;
        auto v = g.getEdgeDst(mid);
        if (v == src) break;
        if (v < src) l = mid + 1;
        else r = mid - 1;
      }
      trans_scores[mid] = scores[e];
    }
  }
}

template <typename DType>
__global__ void compute_alpha_grad_naive(size_t n, int len, Graph g, DType eps, DType scale, DType attn_drop, 
                                         const DType* temp_scores, const DType* norm_scores, 
                                         const mask_t *attn_masks, const DType* feat_in, DType*norm_scores_grad,
                                         DType* scores_grad, DType* alpha_lgrad, DType* alpha_rgrad) {
  DType sum_l[MAX_NUM_CLASSES], sum_r[MAX_NUM_CLASSES];
  for (int i = 0; i < len; i++) sum_l[i] = 0.;
  for (int i = 0; i < len; i++) sum_r[i] = 0.;
  __syncthreads();
  CUDA_KERNEL_LOOP(src, n) {
    auto begin = g.edge_begin(src);
    auto end = g.edge_end(src);
    auto deg_src = g.getDegree(src);
    if (attn_drop > 0.) 
      d_dropout(deg_src, scale, &norm_scores_grad[begin], &attn_masks[begin], &norm_scores_grad[begin]);
    d_softmax((int)deg_src, &norm_scores[begin], &norm_scores_grad[begin], scores_grad);
    DType src_score_grad = 0;
    for (auto e = begin; e != end; e++) {
      auto dst = g.getEdgeDst(e);
      //d_leaky_relu(eps, scores_grad[e-begin], temp_scores[e], temp_score_grad);
      DType temp_score_grad = scores_grad[e-begin] * (temp_scores[e] > 0 ? 1.0 : eps);
      for (int i = 0; i < len; i++) sum_r[i] += feat_in[dst*len+i] * temp_score_grad;
      src_score_grad += temp_score_grad;
    }
    for (int i = 0; i < len; i++) sum_l[i] += feat_in[src*len+i] * src_score_grad;
  }
  for (int i = 0; i < len; ++i) {
    atomicAdd(alpha_lgrad+i, sum_l[i]);
    atomicAdd(alpha_rgrad+i, sum_r[i]);
  }
}

template <typename DType>
__global__ void compute_alpha_grad_warp(size_t n, int len, Graph g, DType eps, DType scale, DType attn_drop, 
                                        const DType* temp_scores, const DType* norm_scores, 
                                        const mask_t *attn_masks, const DType* feat_in, DType*norm_scores_grad,
                                        DType* scores_grad, DType* alpha_lgrad, DType* alpha_rgrad) {
  assert(len <= MAX_NUM_CLASSES);
  __shared__ index_t ptrs[BLOCK_SIZE / WARP_SIZE][2];
  const int thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x; // global thread index
  const int thread_lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
  const int warp_id   = thread_id / WARP_SIZE;   // global warp index
  const int warp_lane = threadIdx.x / WARP_SIZE; // warp index within the CTA
  const int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x; // total number of active warps

  __shared__ DType sum_l[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES], sum_r[BLOCK_SIZE / WARP_SIZE][MAX_NUM_CLASSES];
  for (int i = 0; i < MAX_NUM_CLASSES; i+=WARP_SIZE) if (thread_lane+i < len) sum_l[warp_lane][thread_lane+i] = 0.;
  for (int i = 0; i < MAX_NUM_CLASSES; i+=WARP_SIZE) if (thread_lane+i < len) sum_r[warp_lane][thread_lane+i] = 0.;

  for (int src = warp_id; src < n; src += num_warps) {
    if (thread_lane < 2)
      ptrs[warp_lane][thread_lane] = g.edge_begin(src + thread_lane);
    __syncthreads();
    auto begin = ptrs[warp_lane][0];
    auto end   = ptrs[warp_lane][1];
    auto deg_src = g.getDegree(src);
    //if (attn_drop > 0.) {
    //  for (int i = 0; i < deg_src; i += WARP_SIZE) {
    //    if (thread_lane+i < deg_src) 
    //      norm_scores_grad[begin+thread_lane+i] = norm_scores_grad[begin+thread_lane+i] * attn_masks[begin+thread_lane+i] * scale;
    //  }
    //}
    DType score_sum = 0.;
    for (int i = 0; i < deg_src; i += WARP_SIZE) {
      if (thread_lane+i < deg_src)
        score_sum += norm_scores_grad[begin+thread_lane+i]*norm_scores[begin+thread_lane+i];
    }
    score_sum += SHFL_DOWN(score_sum, 16);
    score_sum += SHFL_DOWN(score_sum, 8);
    score_sum += SHFL_DOWN(score_sum, 4);
    score_sum += SHFL_DOWN(score_sum, 2);
    score_sum += SHFL_DOWN(score_sum, 1);
    score_sum = SHFL(score_sum, 0);
    for (int i = 0; i < deg_src; i += WARP_SIZE) {
      if (thread_lane+i < deg_src) {
        DType pi = norm_scores[begin+thread_lane+i];
        DType dpi = norm_scores_grad[begin+thread_lane+i];
        DType x = pi * (1.0 - pi) * dpi;
        DType y = score_sum - pi*dpi;
        scores_grad[begin+thread_lane+i] = x - y * pi;
      }
    }
    DType src_score_grad = 0;
    for (auto e = begin; e < end; e++) {
      auto dst = g.getEdgeDst(e);
      DType temp_score_grad = scores_grad[e] * (temp_scores[e] > 0 ? 1.0 : eps);
      for (int i = 0; i < len; i += WARP_SIZE) {
        if (thread_lane+i < len)
          sum_r[warp_lane][thread_lane+i] += feat_in[dst*len+thread_lane+i] * temp_score_grad;
      }
      src_score_grad += temp_score_grad;
    }
    for (int i = 0; i < len; i += WARP_SIZE) {
      if (thread_lane+i < len)
        sum_l[warp_lane][thread_lane+i] += feat_in[src*len+thread_lane+i] * src_score_grad;
    }
  }
  // intra-warp reduction
  for (int i = 0; i < len; i += WARP_SIZE) {
    if (thread_lane+i < len) {
      atomicAdd(alpha_lgrad+thread_lane+i, sum_l[warp_lane][thread_lane+i]);
      atomicAdd(alpha_rgrad+thread_lane+i, sum_r[warp_lane][thread_lane+i]);
    }
  }
}

template <typename DType>
void reduce_vertex_sage(size_t nv, int len, Graph g, const DType* in, DType* out, bool trans) {
  update_all_sage<DType><<<(nv - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(nv, len, g, in, out, trans);
  //CudaTest("solving update_all_sage kernel failed");
}

template <typename DType>
void reduce_vertex(size_t nv, int len, Graph g, const DType* scores, const DType* in, DType* out) {
  reduce_warp<DType><<<(nv - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(nv, len, g, scores, in, out);
  //CudaTest("solving reduce kernel failed");
}

template <typename DType>
void compute_scores_grad(size_t nv, int len, Graph g, const DType* feat, const DType* grad, DType* out) {
  compute_scores_grad_warp<DType><<<(nv - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(nv, len, g, feat, grad, out);
  //CudaTest("solving compute_scores_grad kernel failed");
}

template <typename DType>
void compute_scores(size_t nv, int len, Graph g, DType eps, DType scale,
                    DType drop_rate, const DType* alpha_l, const DType* alpha_r, 
                    DType* scores, DType* temp_scores, DType* norm_scores, 
                    DType* rands, mask_t *masks, const DType* in) {
  compute_attn_score_warp<DType><<<(nv - 1) / WARPS_PER_BLOCK + 1, BLOCK_SIZE>>>(nv, len, g, eps, scale, drop_rate,
  //compute_attn_score_naive<DType><<<(nv - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(nv, len, g, eps, scale, drop_rate,
      alpha_l, alpha_r, scores, temp_scores, norm_scores, rands, masks, in);
  //CudaTest("solving compute_scores kernel failed");
}

template <typename DType>
void symmetric_csr_transpose(size_t nv, Graph g, const DType* scores, DType* trans_scores) {
  symmetric_csr_transpose_kernel<DType><<<(nv - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(nv, g, scores, trans_scores);
  //CudaTest("solving symmetric_csr_transpose kernel failed");
}

