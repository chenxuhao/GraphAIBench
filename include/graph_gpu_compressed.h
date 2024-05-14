#pragma once

#include "graph_gpu.h"
#include "cgr_decoder.cuh"
#include "vbyte_decoder.cuh"
#include <curand.h>
#include <curand_kernel.h>

class GraphGPUCompressed : public GraphGPU {
 private:
  std::string scheme;           // compression scheme   
  vidType degree_threshold;     // threshold for hybrid scheme
  eidType *d_rowptr_compressed; // row pointers of Compressed Graph Representation (CGR)
  vidType *d_colidx_compressed; // column induces of Compressed Graph Representation (CGR)

 public:
  GraphGPUCompressed() :
    GraphGPU(),
    scheme(""),
    degree_threshold(32),
    d_rowptr_compressed(NULL),
    d_colidx_compressed(NULL) {
  }
  GraphGPUCompressed(Graph &g, std::string scheme_name, vidType deg=32, int n=0, int m=1, bool unified_mem=false)
    // : GraphGPU(n, m, g.V(), g.E(), g.get_vertex_classes(), g.get_edge_classes(), false, false, g.get_max_degree()) 
  {
    scheme = scheme_name;
    degree_threshold = deg;
    if (unified_mem) unified_init(g);
    else init(g);
  }
  GraphGPUCompressed(bool use_uva, Graph &g, vidType first_v, vidType nv, eidType ne, std::string out_file="none", vidType max_deg=32, std::string scheme_name="streamvbyte") {
    if (out_file != "none") {
      std::string meta_file = out_file + ".meta.txt";
      ifstream f(meta_file.c_str());
      if (f.good()) {
        std::cout << "file " << out_file << " already exists! skipping...\n";
        return;
      }
    }
    scheme = scheme_name;
    degree_threshold = max_deg;
    init_low_sub(g, first_v, ne, nv, use_uva);
    if (out_file != "none") {
      write_to_file(out_file, ne, nv, max_deg);
      CUDA_SAFE_CALL(cudaFree(d_rowptr_compressed));
      CUDA_SAFE_CALL(cudaFree(d_colidx_compressed));
    }
  }
  GraphGPUCompressed(vidType first_v, vidType last_v, vidType max_deg, Graph &g, vidType pref_interval, bool use_uva, std::string out_file="none", std::string scheme_name="streamvbyte") {
    if (out_file != "none") {
      std::string meta_file = out_file + ".meta.txt";
      ifstream f(meta_file.c_str());
      if (f.good()) {
        std::cout << "file " << out_file << " already exists! skipping...\n";
        return;
      }
    }
    scheme = scheme_name;
    degree_threshold = max_deg;
    size_t ne = init_med_sub(g, first_v, last_v, max_deg, pref_interval, use_uva);
    size_t nv = last_v - first_v;
    if (out_file != "none") {
      write_to_file(out_file, ne, nv, max_deg);
      CUDA_SAFE_CALL(cudaFree(d_rowptr_compressed));
      CUDA_SAFE_CALL(cudaFree(d_colidx_compressed));
    }
  }

  void write_to_file(std::string outfilename, size_t ne, size_t nv, vidType max_deg) {
    std::cout << "Writing graph to file\n";
    std::ofstream outfile((outfilename+".vertex.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    std::cout << "idk\n";
    eidType *rowptrs = new eidType[nv+1];
    std::cout << "maybe\n";
    CUDA_SAFE_CALL(cudaMemcpy(rowptrs, d_rowptr_compressed, (nv+1) * sizeof(eidType), cudaMemcpyDeviceToHost));
    outfile.write(reinterpret_cast<const char*>(rowptrs), (nv+1)*sizeof(eidType));
    outfile.close();
    delete [] rowptrs;

    std::ofstream outfile1((outfilename+".edge.bin").c_str(), std::ios::binary);
    if (!outfile1) {
      std::cout << "File not available\n";
      throw 1;
    }
    std::cout << "or here>\n";
    vidType *colidxs = new vidType[ne];
    std::cout << "here?\n";
    CUDA_SAFE_CALL(cudaMemcpy(colidxs, d_colidx_compressed, ne * sizeof(vidType), cudaMemcpyDeviceToHost));
    outfile1.write(reinterpret_cast<const char*>(colidxs), (ne)*sizeof(vidType));
    outfile1.close();
    delete [] colidxs;

    std::ofstream outfile2((outfilename+".meta.txt").c_str(), std::ios::binary);
    if (!outfile1) {
      std::cout << "File not available\n";
      throw 1;
    }
    std::string meta = std::to_string(nv) + "\n" + std::to_string(ne) + "\n";
    meta += "4 8 1 4\n";
    meta += std::to_string(max_deg) + "\n";
    meta += "0\n0\n0\n";
    outfile2 << meta;
    outfile2.close();
    std::cout << "File name " << outfilename << std::endl;
  }

  void init(Graph &hg);
  void unified_init(Graph &hg);
  void init_low_sub(Graph &base_g, vidType first_v, eidType ne, vidType nv, bool use_uva);
  size_t init_med_sub(Graph &hg, vidType first, vidType last, vidType max_deg, vidType pref_interval, bool use_uva);
  inline __device__ vidType read_degree(vidType v) const { return d_degrees[v]; }
  inline __device__ vidType get_degree(vidType v) const {
    if (v == vidType(0-1)) return 0;
    auto start = d_rowptr_compressed[v];
    return d_colidx_compressed[start];
  }
  inline __device__ void run_through(int total_threads) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= total_threads) {
      return;
    }
    vidType nv = sizeof(d_rowptr_compressed) / sizeof(eidType);
    for (vidType v = thread_id; v < nv; v += total_threads) {
      for (eidType e = 0; e < d_rowptr_compressed[v+1]-d_rowptr_compressed[v]; e++) {
        vidType _ = d_colidx_compressed[e];
      }
    }
  }
  inline __device__ vidType warp_decompress(vidType v, vidType *adj) { return decode_cgr_warp<true>(v, adj); }
  inline __device__ vidType* cta_decompress(vidType v, vidType *buf1, vidType *buf2, vidType &degree);
  inline __device__ vidType intersect_num_warp_compressed(vidType v, vidType u, vidType *v_residuals, vidType *u_residuals);
  inline __device__ vidType intersect_num_warp_compressed(vidType u, vidType *adj_u, vidType deg_v, vidType *adj_v);
  inline __device__ vidType cta_intersect_compressed(vidType v, vidType u, vidType *buf1, vidType *buf2, vidType *buf3);
  inline __device__ vidType cta_intersect_compressed(vidType *adj_v, vidType v_degree, vidType *adj_u, vidType u_degree) {
    return intersect_num_cta(adj_v, v_degree, adj_u, u_degree);
  }
  inline __device__ vidType cta_intersect_compressed(vidType v, vidType *buf1, vidType *buf2, vidType u_degree, vidType *adj_u) {
    vidType v_degree = 0;
    vidType *adj_v = cta_decompress(v, buf1, buf2, v_degree);
    return intersect_num_cta(adj_v, v_degree, adj_u, u_degree);
  }
  // decompress CGR format to an (unordered/ordered) vertex set using a warp
  template <bool use_segment>
  inline __device__ vidType decode_cgr_warp(vidType v, vidType *adj) {
    vidType degree = 0;
    if constexpr (use_segment) {
      cgr_decoder_gpu decoder(v, d_colidx_compressed, d_rowptr_compressed[v], adj);
      degree = decoder.decode();
      #ifdef NEED_SORT
      adj = warp_sort(degree, adj, buf); // need a buffer for sorting the vertex set
      #endif
    } else {
      degree = decode_unary_warp_naive(v, adj);
    }
    return degree;
  }
  template <bool use_segment>
  inline __device__ vidType decode_cgr_naive(vidType v, curandState &state) {
    vidType neighbor = decode_unary_naive(v, state);
    return neighbor;
  }
  template <int align_bits>
  inline __device__ vidType decode_unary_segmented_warp(vidType v, vidType *adj) {
    cgr_decoder_gpu decoder(v, d_colidx_compressed, d_rowptr_compressed[v], adj, align_bits);
    return decoder.decode();
  }
  // decode residuals without segment
  inline __device__ vidType decode_unary_warp_naive(vidType v, vidType* out) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
    vidType degree = 0;
    auto begin = d_rowptr_compressed[v];
    auto end = d_rowptr_compressed[v+1];
    if (begin == end) return 0;
    auto in = &d_colidx_compressed[0];
    if (thread_lane == 0) {
      UnaryDecoderGPU decoder(in, begin);
      // decode the first element
      vidType x = decoder.decode_residual_code();
      out[0] = (x & 1) ? v - (x >> 1) - 1 : v + (x >> 1);
      // decode the rest of elements one-by-one
      vidType i = 1;
      while (decoder.get_offset() < end) {
        out[i] = out[i-1] + decoder.decode_residual_code() + 1;
        i++;
      }
      degree = i;
    }
    degree = __shfl_sync(FULL_MASK, degree, 0);
    return degree;
  }
  // decode residuals without segment
  inline __device__ vidType decode_unary_naive(vidType v, curandState &local_state) {
    vidType curr_n;
    auto begin = d_rowptr_compressed[v];
    auto end = d_rowptr_compressed[v+1];
    if (begin == end) return 0;
    auto in = &d_colidx_compressed[0];
    UnaryDecoderGPU decoder(in, begin);
    // decode the first element
    vidType x = decoder.decode_residual_code();
    vidType v_deg = (x & 1) ? v - (x >> 1) - 1 : v + (x >> 1);
    if (v_deg == 0) return vidType(0 - 1);
    vidType n_idx = (vidType)(ceil(curand_uniform(&local_state) * v_deg) - 1);
    // decode the rest of elements one-by-one
    x = decoder.decode_residual_code();
    curr_n = (x & 1) ? v - (x >> 1) - 1 : v + (x >> 1);
    vidType i = 0;
    while (i < n_idx) {
      curr_n += decoder.decode_residual_code() + 1;
      i++;
    }
    // printf("old_t: %d; deg %d; n_idx: %d; new_t: %d\n", v, v_deg, i, curr_n);
    return curr_n;
  }
  // for hybrid scheme where the degree is known
  inline __device__ void decode_unary_warp_naive(vidType v, vidType* out, vidType degree) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
    if (thread_lane == 0) {
      auto offset = d_rowptr_compressed[v] * 32; // transform word-offset to bit-offset
      auto in = &d_colidx_compressed[0];
      UnaryDecoderGPU decoder(in, offset);
      // decode the first element
      vidType x = decoder.decode_residual_code();
      out[0] = (x & 1) ? v - (x >> 1) - 1 : v + (x >> 1);
      // decode the rest of elements one-by-one
      for (vidType i = 1; i < degree; i++) {
        out[i] = out[i-1] + decoder.decode_residual_code() + 1;
      }
    }
  }
  inline __device__ void decode_unary_warp(vidType v, vidType* out, vidType degree) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
    int warp_lane   = threadIdx.x / WARP_SIZE;
    eidType offset = d_rowptr_compressed[v] * 32; // transform word-offset to bit-offset
    // decode the first element
    UnaryDecoderGPU decoder1(d_colidx_compressed, offset);
    if (thread_lane == 0) {
      vidType val = decoder1.decode_residual_code();
      out[0] = (val & 1) ? v - (val >> 1) - 1 : v + (val >> 1);
    }
    offset = __shfl_sync(FULL_MASK, decoder1.get_offset(), 0);;
    if (degree == 1) return;
    //__syncwarp();
 
    // decode the rest of elements in parallel
    //offset += length[warp_lane];
    //int64_t num_bits = d_rowptr_compressed[v+1] * 32 - offset;
    //assert(num_bits > 0);
    //int nrounds = (num_bits - 1) / WARP_SIZE + 1;
    int num_decoded = 1;
    __shared__ int poss[BLOCK_SIZE]; // position is the bit offset, range in [0, 31]
    __shared__ int flags[BLOCK_SIZE]; // whether it is valid
    //__shared__ int64_t prefix_offset[WARPS_PER_BLOCK];
    typedef cub::WarpScan<vidType> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[WARPS_PER_BLOCK];
    int shm_off = warp_lane * WARP_SIZE;
    // each round processes 32 bits of data
    //for (int i = 0; i < nrounds; i ++) {
    while (true) {
      // each thread takes one bit to start decoding; some are invalid
      UnaryDecoderGPU decoder(d_colidx_compressed, offset+thread_lane);
      vidType val = decoder.decode_residual_code() + 1;
      poss[threadIdx.x] = decoder.get_offset() - offset;
      flags[threadIdx.x] = thread_lane ? 0 : 1;
      __syncwarp();
      bool is_active = true;
      // find all valid starting bits within 32 bits
      do {
        if (poss[threadIdx.x] < WARP_SIZE) {
          if (flags[threadIdx.x]) flags[shm_off+poss[threadIdx.x]] = 1;
          poss[threadIdx.x] = poss[threadIdx.x]<32 ? poss[shm_off+poss[threadIdx.x]] : WARP_SIZE;
        }
        is_active = flags[threadIdx.x] && poss[threadIdx.x] < WARP_SIZE;
      } while (__any_sync(0xFFFFFFFF, is_active)); // active thread?
      unsigned mask = __ballot_sync(0xFFFFFFFF, flags[threadIdx.x]);
      // Note that flags[0] = 1, so __popc >= 1
      int valid_index = __popc(mask << (WARP_SIZE-thread_lane-1)) - 1; // index to valid elements (i.e. flag==1)
      int total = __popc(mask); // total number of valid elements decoded in this round
      auto last_valid = 31 - __clz(mask); // mask is definitely not zero

      // write valid elements back
      if (!flags[threadIdx.x]) val = 0;
      if (thread_lane == 0) val += out[num_decoded-1];
      vidType out_val = 0;
      WarpScan(temp_storage[warp_lane]).InclusiveSum(val, out_val);
      // delta coding
      if (flags[threadIdx.x] && valid_index+num_decoded < degree)
        out[valid_index+num_decoded] = out_val;
      if (total + num_decoded >= degree) break;
      num_decoded += total;
      // update the offset based on the last valid element
      //if (thread_lane == last_valid)
      //  prefix_offset[warp_lane] = decoder.get_offset();
      //__syncwarp();
      //offset = prefix_offset[warp_lane];
      offset = __shfl_sync(FULL_MASK, decoder.get_offset(), last_valid);
    }
  }
  inline __device__ vidType warp_decompress(vidType v, vidType *adj, vidType &num_itv, vidType &num_res) {
    cgr_decoder_gpu decoder(v, d_colidx_compressed, d_rowptr_compressed[v]);
    vidType degree = 0;
#ifdef USE_INTERVAL
    degree += decoder.decode_intervals_warp(adj, num_itv);
#endif
    num_res = decoder.decode_residuals_warp(adj+num_itv*2);
    degree += num_res;
    return degree;
  }
  // decompress VByte format to an ordered vertex set using a warp
  template <int scheme = 0, bool delta = true, int pack_size = WARP_SIZE>
  inline __device__ vidType decode_vbyte_warp(vidType v, vidType *adj) {
    auto start = d_rowptr_compressed[v];
    auto length = d_rowptr_compressed[v+1] - start;
    vidType degree = 0;
    if constexpr (scheme == 0) {
      degree = decode_streamvbyte_warp1<delta>(&d_colidx_compressed[start], adj);
    } else {
      degree = decode_varintgb_warp<delta,pack_size>(length, &d_colidx_compressed[start], adj);
    }
    return degree;
  }

  template <int scheme = 0, bool delta = true, int pack_size = WARP_SIZE>
  inline __device__ void decode_vbyte_sums(vidType v, vidType *adj, int n) {
    auto start = d_rowptr_compressed[v];
    decode_streamvbyte_sums1(&d_colidx_compressed[start], adj, n);
  }

  template <int scheme = 0, bool delta = true, int pack_size = WARP_SIZE>
  inline __device__ vidType decode_vbyte_sums(vidType v, int n) {
    auto start = d_rowptr_compressed[v];
    return decode_streamvbyte_sums(&d_colidx_compressed[start], n);
  }

  template <int scheme = 0, bool delta = true, int pack_size = WARP_SIZE>
  inline __device__ vidType decode_vbyte_prefix(vidType v, int n, vidType interval) {
    auto start = d_rowptr_compressed[v];
    return decode_streamvbyte_prefix(&d_colidx_compressed[start], n, interval);
  }

  template <int scheme = 0, bool delta = true, int pack_size = WARP_SIZE>
  inline __device__ vidType decode_1warp(vidType v, vidType *adj, vidType prefix, vidType prefix_bit, int num) {
    auto start = d_rowptr_compressed[v];
    vidType bytes = decode_1vbyte_warp<delta>(&d_colidx_compressed[start], adj, prefix, prefix_bit, num);
    return bytes;
  }

  // decompress VByte format to an ordered vertex set using a warp
  template <int scheme = 0, bool delta = true, int pack_size = WARP_SIZE>
  inline __device__ void decode_vbyte_warp(vidType v, vidType *adj, vidType degree) {
    auto start = d_rowptr_compressed[v];
    if constexpr (scheme == 0) {
      decode_streamvbyte_warp<delta>(degree, &d_colidx_compressed[start], adj);
    } else {
      //decode_varintgb_warp<delta,pack_size>(&d_colidx_compressed[start], adj);
    }
  }
  // decompress VByte format to an ordered vertex set using a warp
  template <int scheme = 0, bool delta = true, int pack_size = WARP_SIZE>
  inline __device__ void decode_vbyte_warp(vidType v, vidType degree, vidType *adj) {
    auto start = d_rowptr_compressed[v];
    decode_streamvbyte_warp<delta>(degree, &d_colidx_compressed[start], adj);
  }
  template <bool use_segment = true>
  inline __device__ vidType decode_hybrid_warp(vidType v, vidType *adj) {
    assert(v < num_vertices);
    auto degree = read_degree(v);
    if (degree == 0) return 0;
    if (degree > degree_threshold) {
      decode_vbyte_warp<0,true>(v, adj, degree);
    } else {
      if constexpr (use_segment) {
        decode_unary_segmented_warp<32>(v, adj); // word_aligned
      } else {
        //decode_unary_warp(v, adj, degree);
        decode_unary_warp_naive(v, adj, degree);
      }
    }
    return degree;
  }

  /*
  inline __device__ vidType intersect_num_warp_compressed(vidType u,
                                                          vidType *adj_u,
                                                          vidType *adj_v,
                                                          vidType deg_v,
                                                          vidType num_itv_v,
                                                          vidType num_res_v);
  inline __device__ vidType intersect_num_warp_compressed_hybrid(vidType* adj_v,
                                                                 vidType deg_v,
                                                                 vidType num_itv_v,
                                                                 vidType num_res_v,
                                                                 vidType* adj_u);
  */
  inline __device__ vidType intersect_num_warp_compressed_hybrid(vidType* adj_v,
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
  // adj_u is to be filled; adj_v is a hybrid set with intervals and residuals
  inline __device__ vidType intersect_num_warp_compressed(vidType u,
                                                          vidType *adj_u,
                                                          vidType *adj_v,
                                                          vidType deg_v,
                                                          vidType num_itv_v,
                                                          vidType num_res_v) {
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
};
