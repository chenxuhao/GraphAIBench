#include "graph_gpu_compressed.h"
#include "vbyte_encoder.hh"

void GraphGPUCompressed::init(Graph &hg) {
  // GraphGPU::init(hg);
  auto nv = hg.num_vertices();
  if (hg.is_compressed()) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr_compressed, (nv+1) * sizeof(eidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_rowptr_compressed, hg.rowptr_compressed(), (nv+1) * sizeof(eidType), cudaMemcpyHostToDevice));
    
    auto len = hg.get_compressed_colidx_length();
    std::cout << "Number of words in compressed edges: " << len << "\n";
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx_compressed, (len+2) * sizeof(uint32_t))); // allocate two more word for memory safty
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

void GraphGPUCompressed::unified_init(Graph &hg) {
  std::cout << "LOADING COMPRESSED GRAPH INTO UNIFIED MEMORY\n" << std::flush;
  auto nv = hg.num_vertices();
  if (hg.is_compressed()) {
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_rowptr_compressed, (nv + 1) * sizeof(eidType)));
    auto compressed_rowptr = hg.rowptr_compressed();
    for (int i = 0; i < nv + 1; i++) {
      d_rowptr_compressed[i] = compressed_rowptr[i];
    }    
    auto len = hg.get_compressed_colidx_length();
    std::cout << "Number of words in compressed edges: " << len << "\n" << std::flush;
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_colidx_compressed, (len+2) * sizeof(uint32_t))); // allocate two more word for memory safty
    auto compressed_colidx = hg.colidx_compressed();
    for (int i = 0; i < len; i++) {
      d_colidx_compressed[i] = compressed_colidx[i];
    }

    if (scheme == "hybrid") {
      const vidType *h_degrees = hg.get_degrees_ptr();
      assert (h_degrees);
      assert (d_degrees == NULL);
      CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_degrees, nv * sizeof(vidType)));
      for (int i = 0; i < nv * sizeof(vidType); i++) {
        d_degrees[i] = h_degrees[i];
      }
    }
  }
}

void GraphGPUCompressed::init_low_sub(Graph &base_g, vidType first_v, eidType ne, vidType nv, bool use_uva) {
  std::cout << "Allocating GPU memory for the low degree subgraph |V| " << nv << " |E| " << ne << "..." << std::endl;
  eidType *_rowptr = new eidType[nv + 1];
  eidType *base_rowptr = base_g._rowptr_compressed() + first_v;
  vidType *base_colidx = base_g._colidx_compressed() + base_rowptr[0];
  eidType diff = base_rowptr[0];
  for (vidType i = 1; i <= nv; i++) {
    _rowptr[i] = base_rowptr[i] - diff;
  }
  if (!use_uva) {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx_compressed, ne * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_colidx_compressed, base_colidx, ne * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr_compressed, (nv+1) * sizeof(eidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_rowptr_compressed, _rowptr, (nv+1) * sizeof(eidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  } else {
    std::cout << "Moving low subgraph onto unified virtual memory...\n";
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_colidx_compressed, ne * sizeof(vidType)));
    for (uint64_t e = 0; e < ne; e++) {
      d_colidx_compressed[e] = base_colidx[e];
    }
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_rowptr_compressed, (nv+1) * sizeof(eidType)));
    for (uint64_t v = 0; v <= nv; v++) {
      d_rowptr_compressed[v] = _rowptr[v];
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());  
  }
  std::cout << "Done" << std::endl;
}

size_t GraphGPUCompressed::init_med_sub(Graph &hg, vidType first, vidType last, vidType m_deg, vidType interval, bool use_uva) {
  vidType nv = last - first;
  vidType interval_key_len = (interval * 2) / 32;
  std::cout << "Allocating GPU memory for the medium degree subgraph |V| " << nv << "..." << std::endl;
  vector<vidType> edges_compressed;
  eidType *vertices_compressed = new eidType[nv+1];
  vertices_compressed[0] = 0;
  vidType *in_buffer = new vidType[m_deg];
  vector<vidType> out_buffer;
  vidType *out_ptr;
  vidType *key_ptr;
  vbyte_encoder vb_encoder("streamvbyte");
  vidType store_interval = min(nv, 10000000);

  size_t ne = 0;
  for (vidType v = first; v < last; v++) {
    if (v % 1000 == 0) std::cout << v << std::endl;
    vidType relabel_v = v - first;
    vidType deg = hg.decode_vertex_vbyte(v, in_buffer, "streamvbyte");
    edges_compressed.push_back(deg);
    if (out_buffer.size() < deg + 1024) out_buffer.resize(deg + 1024);
    uint32_t key_len8 = (deg + 3) / 4;
    uint32_t key_len32 = (key_len8 + 3) / 4;
    key_ptr = out_buffer.data();
    out_ptr = key_ptr + key_len32;
    vidType n_rounds = deg / interval;
    vidType prefix_size = 0;
    vidType total_prefix = 0;

    for (vidType r = 0; r < n_rounds; r++) {
      edges_compressed.push_back(total_prefix);
      prefix_size = vb_encoder.encode(interval, in_buffer, key_ptr, out_ptr);
      total_prefix += prefix_size;
      in_buffer += interval;
      key_ptr += interval_key_len;
      out_ptr += prefix_size;
    }
    vidType count = deg % interval;
    if (count != 0) {
      edges_compressed.push_back(total_prefix);
      prefix_size = vb_encoder.encode(count, in_buffer, key_ptr, out_ptr);
      // total_prefix += prefix_size;
      out_ptr += prefix_size;
    } 
    vidType total_size_v = out_ptr - out_buffer.data();
    edges_compressed.insert(edges_compressed.end(), out_buffer.begin(), out_buffer.begin() + total_size_v);
    if (relabel_v % store_interval == 0 || v == last - 1) {
      size_t _ne = size_t(edges_compressed.size());
      // std::cout << "v " << v << " _ne " << _ne << std::endl;
      ne += _ne;
      edges_compressed.clear();
    }
    in_buffer -= (deg - count);
  }
  std::cout << "Subgraph Colidxs size: " << ne * sizeof(vidType) << "; Rowptrs size: " << (nv + 1) * sizeof(eidType) << "; |V| " << nv << std::endl;
  if (use_uva) {
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_colidx_compressed, ne * sizeof(vidType)));
  } else {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx_compressed, ne * sizeof(vidType)));
  }

  std::cout << "Allocated!\n";

  for (vidType v = first; v < last; v++) {
    vidType relabel_v = v - first;
    vidType deg = hg.decode_vertex_vbyte(v, in_buffer, "streamvbyte");
    edges_compressed.push_back(deg);
    if (out_buffer.size() < deg + 1024) out_buffer.resize(deg + 1024);
    uint32_t key_len8 = (deg + 3) / 4;
    uint32_t key_len32 = (key_len8 + 3) / 4;
    key_ptr = out_buffer.data();
    out_ptr = key_ptr + key_len32;
    vidType n_rounds = deg / interval;
    vidType prefix_size = 0;
    vidType total_prefix = 0;

    for (vidType r = 0; r < n_rounds; r++) {
      edges_compressed.push_back(total_prefix);
      prefix_size = vb_encoder.encode(interval, in_buffer, key_ptr, out_ptr);
      total_prefix += prefix_size;
      in_buffer += interval;
      key_ptr += interval_key_len;
      out_ptr += prefix_size;
    }
    vidType count = deg % interval;
    if (count != 0) {
      edges_compressed.push_back(total_prefix);
      prefix_size = vb_encoder.encode(count, in_buffer, key_ptr, out_ptr);
      // total_prefix += prefix_size;
      out_ptr += prefix_size;
    } 
    vidType total_size_v = out_ptr - out_buffer.data();
    edges_compressed.insert(edges_compressed.end(), out_buffer.begin(), out_buffer.begin() + total_size_v);
    if (relabel_v % store_interval == 0 || v == last - 1) {
      size_t _ne = size_t(edges_compressed.size());
      if (use_uva) {
        for (size_t e = 0; e < _ne; e++) {
          d_colidx_compressed[e] = edges_compressed[e];
        }
      } else {
        CUDA_SAFE_CALL(cudaMemcpy(d_colidx_compressed, &edges_compressed[0], _ne * sizeof(vidType), cudaMemcpyHostToDevice));
      } 
      d_colidx_compressed += _ne;
      edges_compressed.clear();
    }
    vertices_compressed[relabel_v+1] = vertices_compressed[relabel_v] + 1 + ((deg + interval - 1) / interval) + total_size_v;
    in_buffer -= (deg - count);
  }
  // size_t mem_vert = size_t(nv + 1)*sizeof(eidType);
  // size_t mem_edge = size_t(ne)*sizeof(vidType);
  // size_t mem_graph = mem_vert + mem_edge;
  // std::cout << "Med deg subgraph: " << (float)mem_graph / (float)1000000000 << "GB; |V| " << nv << "\n";
  if (use_uva) {
    std::cout << "Moving medium subgraph onto unified virtual memory...\n";
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&d_rowptr_compressed, (nv+1) * sizeof(eidType)));
    for (uint64_t v = 0; v <= nv; v++) {
      d_rowptr_compressed[v] = vertices_compressed[v];
    }
  } else {
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr_compressed, (nv+1) * sizeof(eidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_rowptr_compressed, vertices_compressed, (nv+1) * sizeof(eidType), cudaMemcpyHostToDevice));
  }
  d_colidx_compressed -= ne;
  std::cout << "Done" << std::endl;
  return ne;
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
#ifdef USE_INTERVAL
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
  #ifdef USE_INTERVAL
  auto deg_u = u_decoder.decode_intervals_warp(adj_u, num_itv_u);
  #endif
  auto u_residuals = adj_u + num_itv_u*2;
  auto num_res_u = u_decoder.decode_residuals_warp(u_residuals);
  #ifdef USE_INTERVAL
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
#ifdef USE_INTERVAL
  __shared__ vidType num_itv_v[WARPS_PER_BLOCK], num_itv_u[WARPS_PER_BLOCK];
#endif
  __shared__ vidType num_res_v[WARPS_PER_BLOCK], num_res_u[WARPS_PER_BLOCK];
  if (thread_lane == 0) {
#ifdef USE_INTERVAL
    num_itv_v[warp_lane] = 0;
    num_itv_u[warp_lane] = 0;
#endif
    num_res_v[warp_lane] = 0;
    num_res_u[warp_lane] = 0;
  }
  __syncwarp();
  cgr_decoder_gpu v_decoder(v, d_colidx_compressed, d_rowptr_compressed[v]);
  cgr_decoder_gpu u_decoder(u, d_colidx_compressed, d_rowptr_compressed[u]);
#ifdef USE_INTERVAL
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

#ifdef USE_INTERVAL
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
#ifdef USE_INTERVAL
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

