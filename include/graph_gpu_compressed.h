#include "graph_gpu.h"
#include "cgr_decoder.cuh"
#include "vbyte_decoder.cuh"

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
  GraphGPUCompressed(Graph &g, std::string scheme_name, vidType deg=32, int n=0, int m=1) : 
      GraphGPU(n, m, g.V(), g.E(), g.get_vertex_classes(), g.get_edge_classes()) {
    scheme = scheme_name;
    degree_threshold = deg;
    init(g);
  }
  void init(Graph &hg);
  inline __device__ vidType read_degree(vidType v) const { return d_degrees[v]; }
  //inline __device__ vidType decode_cgr_warp(vidType v, vidType *adj);
  inline __device__ vidType warp_decompress(vidType v, vidType *adj) { return decode_cgr_warp(v, adj); }
  //inline __device__ vidType warp_decompress(vidType v, vidType *adj, vidType &num_itv, vidType &num_res);
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
  inline __device__ vidType decode_cgr_warp(vidType v, vidType *adj) {
    cgr_decoder_gpu decoder(v, d_colidx_compressed, d_rowptr_compressed[v], adj);
    vidType degree = decoder.decode();
    #ifdef NEED_SORT
    adj = warp_sort(degree, adj, buf); // need a buffer for sorting the vertex set
    #endif
    return degree;
  }
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
    auto in = &d_colidx_compressed[0];
    __shared__ int64_t length[WARPS_PER_BLOCK];
    vidType val = 0;
    // decode the first element
    if (thread_lane == 0) {
      UnaryDecoderGPU decoder(in, offset);
      val = decoder.decode_residual_code();
      out[0] = (val & 1) ? v - (val >> 1) - 1 : v + (val >> 1);
      length[warp_lane] = decoder.get_offset() - offset;
    }
    if (degree == 1) return;
    __syncwarp();
    assert(length[warp_lane] > 0);
    if (length[warp_lane] > 36) printf("length=%ld\n", length[warp_lane]);
    offset += length[warp_lane];
    int64_t num_bits = d_rowptr_compressed[v+1] * 32 - offset;
    assert(num_bits > 0);
 
    // decode the rest of elements in parallel
    int nrounds = (num_bits - 1) / WARP_SIZE + 1;
    int num_decoded = 1;
    __shared__ int poss[BLOCK_SIZE];
    __shared__ int flags[BLOCK_SIZE];
    __shared__ int64_t prefix_offset[WARPS_PER_BLOCK];
    typedef cub::WarpScan<vidType> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[WARPS_PER_BLOCK];
    int shm_off = warp_lane * WARP_SIZE;
    int prefix = out[0];
    // each round processes 32 bits of data
    for (int i = 0; i < nrounds; i ++) {
      // each thread takes one bit to start decoding; some are invalid
      UnaryDecoderGPU decoder(in, offset+thread_lane);
      assert (offset < int64_t(117185093)*32);
      //if (offset > int64_t(117185093)*32);
      //  printf("v %d word_offset=%ld, bit_offset=%ld\n", v, (offset+thread_lane)/32, offset+thread_lane);
      val = decoder.decode_residual_code() + 1;
      assert(decoder.get_offset() > offset);
      //if (thread_lane == 0 && decoder.get_offset() - offset >= 32)
      //  printf("v %d round %d thread_lane=%d, old_offset=%ld, new_offset=%ld, num_decoded=%d\n", v, i, thread_lane, offset, decoder.get_offset(), num_decoded);
      //assert(thread_lane != 0 || decoder.get_offset() - offset < 32);
      poss[threadIdx.x] = decoder.get_offset() - offset;
      flags[threadIdx.x] = thread_lane ? 0 : 1;
      __syncwarp();
      int flag = flags[threadIdx.x]; // whether it is valid
      int pos = poss[threadIdx.x]; // position is the bit offset, range in [0, 31]
      bool is_active = true;
      // find all valid starting bits within 32 bits
      do {
        if (pos < WARP_SIZE) {
          if (flag) flags[shm_off+pos] = 1;
          poss[threadIdx.x] = pos<32 ? poss[shm_off+pos] : WARP_SIZE;
        }
        flag = flags[threadIdx.x];
        pos = poss[threadIdx.x];
        is_active = flag && pos < WARP_SIZE;
      } while (__any_sync(0xFFFFFFFF, is_active)); // active thread?
      int valid_index = 0; // index to valid elements (i.e. flag==1)
      int total = 0; // number of elements decoded in this round
      //WarpScan(temp_storage[warp_lane]).ExclusiveSum(flag, valid_index, total);
      unsigned mask = __ballot_sync(0xFFFFFFFF, flag);
      // Note that flag[0] = 1, so __popc >= 1
      valid_index = __popc(mask << (WARP_SIZE-thread_lane-1)) - 1;
      total = __popc(mask);

      // write valid elements back
      if (!flag) val = 0;
      if (thread_lane == 0) val += prefix;
      vidType out_val = 0;
      WarpScan(temp_storage[warp_lane]).InclusiveSum(val, out_val);
      // delta coding
      if (flag && valid_index+num_decoded < degree)
        out[valid_index+num_decoded] = out_val;
      //if (flag) out[num_decoded[warp_lane]+valid_index] = out_val + prefix;
      //if (thread_lane == 0) num_decoded[warp_lane] += __popc(mask);
      if (total + num_decoded >= degree) break;
      num_decoded += total;
      prefix = out[num_decoded-1];
      // update the offset based on the last valid element
      if (thread_lane == 31 - __clz(mask)) // mask is definitely not zero
        prefix_offset[warp_lane] = decoder.get_offset();
      __syncwarp();
      offset = prefix_offset[warp_lane];
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
      degree = decode_streamvbyte_warp<delta>(length, &d_colidx_compressed[start], adj);
    } else {
      degree = decode_varintgb_warp<delta,pack_size>(length, &d_colidx_compressed[start], adj);
    }
    return degree;
  }
  inline __device__ vidType decode_hybrid_warp(vidType v, vidType *adj) {
    auto degree = read_degree(v);
    if (degree == 0) return 0;
    if (degree > degree_threshold) {
      decode_vbyte_warp<0,true>(v, adj);
    } else {
      decode_unary_warp(v, adj, degree);
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
