#pragma once
#include "graph_gpu_compressed.h"
#include "sampling_utils.h"

typedef GraphGPUCompressed GraphTy;
const vidType MAX_VIDTYPE = 0 - 1;


__device__ int sample_size_gpu(int step) {
    if (step == -1) return 1;
    if (step == 0) return 25;
    return 10;
    // if (step == -1) return 2;
    // return 2;
}

template <int scheme = 0, bool delta = true, int pack_size = 4>
__device__ vidType next_gpu(GraphTy &g, vidType transit, int thread_id, vidType *buffer, curandState state) {
    // int warp_id = thread_id / WARP_SIZE;
    vidType max_degree = g.get_max_degree();
    vidType *adj_v = buffer + max_degree * thread_id;
    vidType src_degree = g.decode_vbyte_warp<scheme,delta,pack_size>(transit, adj_v);
    if (src_degree == 0) { return MAX_VIDTYPE; }
    int idx = (int)(ceil(curand_uniform(&state) * src_degree) - 1);
    return adj_v[idx];
}