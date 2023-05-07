#include "graph_gpu_compressed.h"

void GraphGPUCompressed::init(Graph &hg) {
  GraphGPU::init(hg);
  auto nv = hg.num_vertices();
  if (hg.is_compressed()) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr_compressed, (nv+1) * sizeof(eidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_rowptr_compressed, hg.rowptr_compressed(), (nv+1) * sizeof(eidType), cudaMemcpyHostToDevice));
    
    auto len = hg.get_compressed_colidx_length();
    std::cout << "Number of words in compressed edges: " << len << "\n";
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx_compressed, (len+1) * sizeof(uint32_t))); // allocate one more word for memory safty
    CUDA_SAFE_CALL(cudaMemcpy(d_colidx_compressed, hg.colidx_compressed(), len * sizeof(uint32_t), cudaMemcpyHostToDevice));

    if (scheme == "hybrid") {
      const vidType *h_degrees = hg.get_degrees_ptr();
      assert (h_degrees);
      assert (d_degrees == NULL);
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_degrees, nv * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMemcpy(d_degrees, h_degrees, nv * sizeof(vidType), cudaMemcpyHostToDevice));
    }
  }
}
/*
// decompress CGR format to an (unordered/ordered) vertex set using a warp
inline __device__ vidType GraphGPUCompressed::decode_cgr_warp(vidType v, vidType *adj) {
  cgr_decoder_gpu decoder(v, d_colidx_compressed, d_rowptr_compressed[v], adj);
  vidType degree = decoder.decode();
#ifdef NEED_SORT
  adj = warp_sort(degree, adj, buf); // need a buffer for sorting the vertex set
#endif
  return degree;
}

inline __device__ void GraphGPUCompressed::decode_unary_warp(vidType v, vidType* out, vidType degree) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  if (thread_lane == 0) {
    auto offset = d_rowptr_compressed[v] * 32; // transform word-offset to bit-offset
    auto in = &d_colidx_compressed[0];
    UnaryDecoderGPU decoder(in, offset);
    // decode the first element
    vidType x = decoder.decode_residual_code();
    out[0] = (x & 1) ? v - (x >> 1) - 1 : v + (x >> 1);
    // decode the rest of elements
    for (vidType i = 1; i < degree; i++) {
      out[i] = out[i-1] + decoder.decode_residual_code() + 1;
    }
  }
}

// decompress to a hybrid vertex set (intervals+residuals) using a warp
inline __device__ vidType GraphGPUCompressed::warp_decompress(vidType v, vidType *adj, vidType &num_itv, vidType &num_res) {
  cgr_decoder_gpu decoder(v, d_colidx_compressed, d_rowptr_compressed[v]);
  vidType degree = 0;
#if USE_INTERVAL
  degree += decoder.decode_intervals_warp(adj, num_itv);
#endif
  num_res = decoder.decode_residuals_warp(adj+num_itv*2);
  degree += num_res;
  return degree;
}

// adj_u is to be filled; adj_v is a hybrid set with intervals and residuals
inline __device__ vidType GraphGPUCompressed::intersect_num_warp_compressed(vidType u,
                                                                            vidType *adj_u,
                                                                            vidType *adj_v,
                                                                            vidType deg_v,
                                                                            vidType num_itv_v,
                                                                            vidType num_res_v) {
  //int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  vidType num = 0, num_itv_u = 0;
  assert(deg_v >= num_res_v);
  assert(num_itv_v > 0 || deg_v == num_res_v); // if num_itv_v == 0, then deg_v == num_res_v
  auto v_residuals = adj_v + num_itv_v*2;
  cgr_decoder_gpu u_decoder(u, d_colidx_compressed, d_rowptr_compressed[u]);
  #if USE_INTERVAL
  auto deg_u = u_decoder.decode_intervals_warp(adj_u, num_itv_u);
  #endif
  auto u_residuals = adj_u + num_itv_u*2;
  auto num_res_u = u_decoder.decode_residuals_warp(u_residuals);
  #if USE_INTERVAL
  num += intersect_num_itv_itv(num_itv_v, adj_v, num_itv_u, adj_u);
  num += intersect_num_itv_res(num_itv_v, adj_v, num_res_u, u_residuals);
  num += intersect_num_itv_res(num_itv_u, adj_u, num_res_v, v_residuals);
  #endif
  num += intersect_num(v_residuals, num_res_v, u_residuals, num_res_u);
  return num;
}

inline __device__ vidType GraphGPUCompressed::intersect_num_warp_compressed_hybrid(vidType* adj_v,
                                                                                   vidType deg_v,
                                                                                   vidType num_itv_v,
                                                                                   vidType num_res_v,
                                                                                   vidType* adj_u) {
 vidType count = 0;
 for (vidType i = 0; i < num_itv_v; i++) {
   auto len = adj_v[i*2+1];
   for (vidType j = 0; j < len; j++) {
     auto u = adj_v[i*2] + j;
     count += intersect_num_warp_compressed(u, adj_u, adj_v, deg_v, num_itv_v, num_res_v);
   }
 }
 vidType idx = num_itv_v*2;
 for (vidType i = 0; i < num_res_v; i++) {
   auto u = adj_v[idx+i];
   count += intersect_num_warp_compressed(u, adj_u, adj_v, deg_v, num_itv_v, num_res_v);
 }
 return count;
}
*/
// using a CTA to decompress the adj list of a vertex
inline __device__ vidType* GraphGPUCompressed::cta_decompress(vidType v, vidType *buf1, vidType *buf2, vidType &degree) {
  cgr_decoder_gpu decoder(v, d_colidx_compressed, d_rowptr_compressed[v]);
  __shared__ vidType num_items;
  if (threadIdx.x == 0) num_items = 0;
  __syncthreads();
  decoder.decode_intervals_cta(buf1, &num_items);
  decoder.decode_residuals_cta(buf1, &num_items);
  degree = num_items;
  vidType *adj = buf1;
#ifdef NEED_SORT
  adj = cta_sort(num_items, buf1, buf2);
#endif
  return adj;
}

inline __device__ vidType GraphGPUCompressed::intersect_num_warp_compressed(vidType v,
                                                                            vidType u,
                                                                            vidType *v_residuals,
                                                                            vidType *u_residuals) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  vidType num = 0;
#if USE_INTERVAL
  __shared__ vidType num_itv_v[WARPS_PER_BLOCK], num_itv_u[WARPS_PER_BLOCK];
#endif
  __shared__ vidType num_res_v[WARPS_PER_BLOCK], num_res_u[WARPS_PER_BLOCK];
  if (thread_lane == 0) {
#if USE_INTERVAL
    num_itv_v[warp_lane] = 0;
    num_itv_u[warp_lane] = 0;
#endif
    num_res_v[warp_lane] = 0;
    num_res_u[warp_lane] = 0;
  }
  __syncwarp();
  cgr_decoder_gpu v_decoder(v, d_colidx_compressed, d_rowptr_compressed[v]);
  cgr_decoder_gpu u_decoder(u, d_colidx_compressed, d_rowptr_compressed[u]);
#if USE_INTERVAL
  __shared__ vidType v_begins[WARPS_PER_BLOCK][32], v_ends[WARPS_PER_BLOCK][32];
  __shared__ vidType u_begins[WARPS_PER_BLOCK][32], u_ends[WARPS_PER_BLOCK][32];
  auto n_items = v_decoder.decode_intervals_warp(v_begins[warp_lane], v_ends[warp_lane]);
  if (thread_lane == 0) num_itv_v[warp_lane] = n_items;
  n_items = u_decoder.decode_intervals_warp(u_begins[warp_lane], u_ends[warp_lane]);
  if (thread_lane == 0) num_itv_u[warp_lane] = n_items;
  __syncwarp();
  assert(num_itv_v[warp_lane] < 32);
  assert(num_itv_u[warp_lane] < 32);
  //if (thread_lane == 0) printf("v %u has %u intervals, u %u has %u intervals\n", v, num_itv_v[warp_lane], u, num_itv_u[warp_lane]);
#endif
  auto degree = v_decoder.decode_residuals_warp(v_residuals);
  if (thread_lane == 0) num_res_v[warp_lane] = degree;
  degree = u_decoder.decode_residuals_warp(u_residuals);
  if (thread_lane == 0) num_res_u[warp_lane] = degree;

#if USE_INTERVAL
  // compare v_itv and u_itv
  num += intersect_num_itv_itv(num_itv_v[warp_lane], v_begins[warp_lane], v_ends[warp_lane],
      num_itv_u[warp_lane], u_begins[warp_lane], u_ends[warp_lane]);
  // compare v_itv and u_res
  num += intersect_num_itv_res(num_itv_v[warp_lane], v_begins[warp_lane], v_ends[warp_lane],
      num_res_u[warp_lane], u_residuals);
  // compare v_res and u_itv
  num += intersect_num_itv_res(num_itv_u[warp_lane], u_begins[warp_lane], u_ends[warp_lane],
      num_res_v[warp_lane], v_residuals);
#endif
  // compare v_res and u_res
  num += intersect_num(v_residuals, num_res_v[warp_lane], u_residuals, num_res_u[warp_lane]);
  return num;
}

// adj_u is to be filled; adj_v is a sorted vertex set
inline __device__ vidType GraphGPUCompressed::intersect_num_warp_compressed(vidType u,
                                                                            vidType *adj_u,
                                                                            vidType deg_v,
                                                                            vidType *adj_v) {
  //int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  vidType num = 0, num_itv_u = 0, num_res_u = 0;
  cgr_decoder_gpu decoder(u, d_colidx_compressed, d_rowptr_compressed[u]);
#if USE_INTERVAL
  auto deg_u = decoder.decode_intervals_warp(adj_u, num_itv_u);
  num += intersect_num_itv_res(num_itv_u, adj_u, deg_v, adj_v);
#endif
  vidType *u_residuals = adj_u + num_itv_u*2;
  num_res_u = decoder.decode_residuals_warp(u_residuals);
  num += intersect_num(adj_v, deg_v, u_residuals, num_res_u);
  return num;
}

inline __device__ vidType GraphGPUCompressed::cta_intersect_compressed(vidType v,
                                                                       vidType u,
                                                                       vidType *buf1,
                                                                       vidType *buf2,
                                                                       vidType *buf3) {
  vidType count = 0;
  vidType *adj_v, *adj_u, v_degree = 0, u_degree = 0;
  adj_v = cta_decompress(v, buf1, buf2, v_degree);
  if (adj_v == buf2)
    adj_u = cta_decompress(u, buf1, buf3, u_degree);
  else
    adj_u = cta_decompress(u, buf2, buf3, u_degree);
  count = intersect_num_cta(adj_v, v_degree, adj_u, u_degree);
  return count;
} 

