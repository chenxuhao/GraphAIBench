#pragma once
#include "graph_gpu_compressed.h"
#include "sampling_utils.h"
#include <stdio.h>

const vidType MAX_VIDTYPE = 0 - 1;


__device__ int sample_size_gpu(int step) {
    if (step == -1) return 1;
    // if (step == 0) return 25;
    // return 10;
    if (step == 0) return 2;
    return 3;
}

template <int scheme = 1, bool delta = true, int pack_size = 4>
__device__ vidType next_gpu(GraphGPUCompressed &g, vidType transit, int thread_id, vidType *buffer, int max_deg, curandState state) {
    int warp_id = thread_id / WARP_SIZE;
    int num_warps = (256 / WARP_SIZE) * gridDim.x;
    vidType *adj_v = buffer + (max_deg*warp_id);
    vidType src_degree = g.decode_vbyte_warp<scheme,delta,pack_size>(transit, adj_v);
    if (src_degree == 0) { return MAX_VIDTYPE; }
    int idx = (int)(ceil(curand_uniform(&state) * src_degree) - 1);
    printf("idx %d\n", idx);
    return adj_v[idx];
}

template <int scheme = 0, bool delta = true, int pack_size = 4>
__device__ vidType decompress_edge(GraphGPUCompressed &g, vidType transit, int warp_id, vidType *adj_v) {
    vidType src_degree = g.decode_vbyte_warp<scheme,delta,pack_size>(transit, adj_v);
    return src_degree;
}

__device__ vidType next_gpu2(vidType *adj, int degree, curandState state) {
    int idx = (int)(ceil(curand_uniform(&state) * degree) - 1);
    return adj[idx];
}
