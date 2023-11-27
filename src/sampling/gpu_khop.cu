#include "graph_gpu.h"
#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
using namespace std;
using namespace cooperative_groups;

const int BLOCK_DIM = 32;

inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


__device__ __inline__ int next(GraphGPU &g, int transit, int idx, curandState state) {
    // return transit * 10 + idx;
    int n = g.getOutDegree(transit);
    if (n == 0) {
        return transit;
    }
    int chosen = (int)(ceil(curand_uniform(&state) * n) - 1);
    return g.N(transit, chosen);
}

__global__ void baseline(GraphGPU g, int* result, int steps, int* sample_size, int cur_num, int total_threads, int seed) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= total_threads) {
        return;
    }
    int d, r, low, high;
    int prev_begin = 0;
    int begin = cur_num;
    int transit, idx;
    curandState ix_state;
    curand_init(seed, ix, 0, &ix_state);
    grid_group grid = this_grid();
    for (int i = 0; i < steps; i++) {
        cur_num *= sample_size[i];
        d = cur_num / total_threads; 
        r = cur_num % total_threads;
        if (ix < r) {
            low = ix * (d + 1);
            high = (ix + 1) * (d + 1);
        } else {
            low = r + ix * d;
            high = r + ix * d + d;
        }
        for (int j = low; j < high; j++) {
            idx = j % sample_size[i];
            transit = result[j / sample_size[i] + prev_begin];
            result[begin + j] = next(g, transit, idx, ix_state);
        }
        prev_begin = begin;
        begin += cur_num;        
        grid.sync();
        
    }
} 
// 2 * 2 *3

// 1, 2
// 11, 12, 21, 22
// 111, 112, 113, 121, 122, 123, 211, 212, 213, 221, 222, 223
// 40000 * 25 * 10 + 40000 * 25 + 40000
double khop_sample(Graph &g, vector<int>& initial, int steps, int* sample_size, int total_num, int* result, int pdeg=128, int seed=0) {
    GraphGPU gpu_g (g);
    int cur_num = initial.size();
    int *d_result, *d_sample_size;
    int total_threads = pdeg;
    dim3 block(BLOCK_DIM);
    dim3 grid((total_threads + block.x - 1) / block.x);
    double iStart, iElaps, oStart, oElaps = 0;
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }
    
    oStart = seconds();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, total_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_sample_size, steps * sizeof(int)));

    CUDA_SAFE_CALL(cudaMemcpy(d_result, result, cur_num * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_sample_size, sample_size, steps * sizeof(int), cudaMemcpyHostToDevice));

    void *kernel_args[] = {&gpu_g, &d_result, &steps, &d_sample_size, &cur_num, &total_threads, &seed};
    oElaps += seconds() - oStart;
    iStart = seconds();
    cudaLaunchCooperativeKernel((void*)(baseline), grid, block, kernel_args);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    oStart = seconds();
    CUDA_SAFE_CALL(cudaMemcpy(result, d_result, total_num * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_result));
    CUDA_SAFE_CALL(cudaFree(d_sample_size));
    oElaps += seconds() - oStart;

    cout << "Time elapsed for allocating and copying " << oElaps << " sec\n\n";

    return iElaps;
}

// cpu:1.41311e-3
// gpu:
// kernel: 4.19617e-5
// allocating and copying others: 4.68731e-4
// allocating graph: 0.15959
// copying graph: 8.2e-5