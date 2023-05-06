#include "graph_gpu.h"
#include "cgr_decoder.cuh"
#include "vbyte_decoder.cuh"
//#define USE_INTERVAL 1

class GraphGPUCompressed : public GraphGPU {
 private:
  eidType *d_rowptr_compressed;     // row pointers of Compressed Graph Representation (CGR)
  vidType *d_colidx_compressed;     // column induces of Compressed Graph Representation (CGR)

 public:
  GraphGPUCompressed() :
    GraphGPU(),
    d_rowptr_compressed(NULL),
    d_colidx_compressed(NULL) {
  }
  GraphGPUCompressed(Graph &g, int n=0, int m=1) : 
      GraphGPU(n, m, g.V(), g.E(), g.get_vertex_classes(), g.get_edge_classes()) {
    init(g);
  }
  void init(Graph &hg);
  inline __device__ void decode_unary_warp(vidType v, vidType* out, vidType degree);
  inline __device__ vidType decode_cgr_warp(vidType v, vidType *adj);
  inline __device__ vidType warp_decompress(vidType v, vidType *adj) { decode_cgr_warp(v, adj); }
  inline __device__ vidType* cta_decompress(vidType v, vidType *buf1, vidType *buf2, vidType &degree);
  inline __device__ vidType warp_decompress(vidType v, vidType *adj, vidType &num_itv, vidType &num_res);
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
  inline __device__ vidType decompress_adj_warp(vidType v, vidType *adj, int scheme = 0, vidType degree = 0) {
    if (scheme == 0) {
      decode_unary_warp(v, adj, degree);
    } else if (scheme == 1) {
      decode_cgr_warp(v, adj);
    } else {
      decode_vbyte_warp(v, adj);
    }
  }

  // adj_u is to be filled; adj_v is a hybrid set with intervals and residuals
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
};
