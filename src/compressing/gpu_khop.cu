#include "graph.h"
#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include "khop_gpu.cuh"
#include "khop.h"
using namespace std;
using namespace cooperative_groups;

const int BLOCK_DIM = 32;


__device__ __inline__ int next(Graph &g, int transit, int idx, curandState state) {
    // return transit * 10 + idx;
    int n = g.getOutDegree(transit);
    if (n == 0) {
        return transit;
    }
    int chosen = (int)(ceil(curand_uniform(&state) * n) - 1);
    return g.N(transit, chosen);
}

__global__ void khop_sample(Graph g, int* result, int steps, int cur_num, int total_threads, int seed) {
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
        cur_num *= sample_size(i);
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
            idx = j % sample_size(i);
            transit = result[j / sample_size(i) + prev_begin];
            result[begin + j] = next(g, transit, idx, ix_state);
        }
        prev_begin = begin;
        begin += cur_num;        
        grid.sync();
        
    }
} 

// 40000 * 25 * 10 + 40000 * 25 + 40000
double multilayer_sample(Graph &g, vector<vidType>& initial, int steps, int total_num, vidType* result, int pdeg=128, int seed=0) {
    int cur_num = initial.size();
    vidType* d_result;
    int total_threads = pdeg;
    dim3 block(BLOCK_DIM);
    dim3 grid((total_threads + block.x - 1) / block.x);
    Timer alloc_t, sample_t, dealloc_t;
    int size = sizeof(vidType);
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }
    
    alloc_t.Start();
    cudaMalloc((void **)&d_result, total_num * size);

    cudaMemcpy(d_result, result, cur_num * size, cudaMemcpyHostToDevice);

    void *kernel_args[] = {&g, &d_result, &steps, &cur_num, &total_threads, &seed};
    alloc_t.Stop();
    sample_t.Start();
    cudaLaunchCooperativeKernel((void*)(khop_sample), grid, block, kernel_args);
    cudaDeviceSynchronize();
    sample_t.Stop();

    dealloc_t.Start();
    cudaMemcpy(result, d_result, total_num * size, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    dealloc_t.Stop();

    cout << "Time elapsed for allocating and copying " << alloc_t.Seconds() + dealloc_t.Seconds() << " sec\n\n";

    return sample_t.Seconds();
}


int main(int argc, char* argv[]) {
  Graph g;
  std::string in_prefix = argv[1];
  std::string out_prefix = argv[2];
  std::string scheme = "streamvbyte";
  bool permutated = false;
  // save_compressed_graph(in_prefix, out_prefix);
  g.load_compressed_graph(out_prefix, scheme, permutated);
  // g.print_meta_data();
  std::cout << "LOADED COMPRESSED GRAPH\n" << std::endl;

  int n_samples = argc >= 3 ? atoi(argv[3]) : num_samples();
  int pdeg = argc >= 4 ? atoi(argv[4]) : 128;

  double iElaps;
  vector<vidType> initial = get_initial_transits(sample_size(-1) * n_samples, g.V());
  int step_count = steps();
  int total_count = step_count;
  int cur_num = sample_size(-1) * n_samples;
  for (int step = 0; step < steps(); step++) {
    step_count *= sample_size(step);
    total_count += step_count;
  }
  vidType* result = new vidType[total_count];
  iElaps = multilayer_sample(g, initial, step_count, total_count, result, pdeg);

  cout << "Time elapsed " << iElaps << " sec\n\n";
  delete[] result;

  return 0;
}