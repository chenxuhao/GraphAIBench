#pragma once
#include "graph_gpu_compressed.h"
#include "sampling_utils.h"
#include <stdio.h>

const vidType MAX_VIDTYPE = 0 - 1;


__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets a different seed, same sequence
       number, no offset */
    // curand_init(id, 0, 0, &state[id]);
    curand_init(1234, id, 0, &state[id]);
}

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

__device__ vidType get_next_gpu(GraphGPU &g, vidType transit, int deg, curandState state) {
    if (deg == 0) { return MAX_VIDTYPE; }
    eidType idx = (eidType)(ceil(curand_uniform(&state) * deg) - 1);
    return g.N(transit, idx);
}

__device__ void set_random_idxs(GraphGPU &g, int old_t_begin, int t_begin, vidType *result, int *random_idxs, int step_sample_size, int step_count, int total_threads, curandState local_state) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    // sample fan out size num of random indices for single transit per warp
    for (int i = thread_id; i < step_count; i += total_threads) {
        int old_t_idx = old_t_begin + i / step_sample_size;
        vidType old_t = result[old_t_idx];
        vidType old_t_deg = 0;
        if (old_t != MAX_VIDTYPE) {
            old_t_deg = g.get_degree(old_t);
        }
        if (old_t_deg == 0) { // no need to continue sampling indices for 0 degree vertices
            int t_idx = t_begin + i;
            result[t_idx] = MAX_VIDTYPE;
        }
        else {
            random_idxs[i] = (int)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
            // printf("old %d; r[i] %d\n", old_t, random_idxs[i]);
        }
    }
}