#include "graph.h"
#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include "khop_gpu.cuh"
#include "khop.h"
#include "graph_gpu_compressed.h"
using namespace std;
using namespace cooperative_groups;

// const int BLOCK_DIM = 32;
// const vidType MAX_VIDTYPE = 0 - 1;


__global__ void khop_sample(GraphTy g, vidType* result, int step, int t_begin, int old_t_begin, int total_threads, vidType *buffer, int seed) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= total_threads) {
        return;
    }
    curandState ix_state;
    curand_init(seed, thread_id, 0, &ix_state);
    int t_idx = t_begin + thread_id;
    int old_t_idx = old_t_begin + thread_id / sample_size_gpu(step);
    vidType old_t = result[old_t_idx];
    if (old_t == MAX_VIDTYPE) {
        result[t_idx] = MAX_VIDTYPE;
    }
    else {
        vidType new_t = next_gpu(g, old_t, thread_id, buffer, ix_state);
        result[t_idx] = new_t;
    }
} 

// 40000 * 25 * 10 + 40000 * 25 + 40000
double multilayer_sample(Graph &g, vector<vidType>& initial, int n_samples, int total_num, vidType* result, int pdeg=128, int seed=0) {
    GraphGPUCompressed gg(g, "streamvbyte", g.get_degree_threshold());
    int cur_num = initial.size();
    vidType *d_result, *buffer;
    int total_threads = pdeg;
    Timer alloc_t, sample_t, dealloc_t;
    int size = sizeof(vidType);
    // size_t warps_per_block = total_threads / WARP_SIZE;
    // size_t nblocks = (g.V()-1)/warps_per_block+1;
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }

    std::cout << "Allocating buffer for decompressed adjacency lists\n";
    alloc_t.Start();
    allocate_gpu_buffer(size_t(g.get_max_degree()) * total_threads, buffer);
    cudaMalloc((void **)&d_result, total_num * size);
    cudaMemcpy(d_result, result, cur_num * size, cudaMemcpyHostToDevice);
    alloc_t.Stop();

    std::cout << "Starting sampling with " << total_threads << " threads...\n";
    sample_t.Start();
    int step_count = sample_size(-1) * n_samples;
    int prev_step_count = n_samples;
    int t_begin = 0;
    int old_t_begin = 0;
    for (int step = 0; step < steps(); step++) {
        t_begin += step_count;
        step_count *= sample_size(step);
        prev_step_count *= sample_size(step-1);
        int num_blocks = (step_count + total_threads - 1) / total_threads;
        khop_sample<<<num_blocks,total_threads>>>(gg, d_result, step, t_begin, old_t_begin, step_count, buffer, seed);
        cudaDeviceSynchronize();
        old_t_begin += prev_step_count;
    }
    sample_t.Stop();

    dealloc_t.Start();
    cudaMemcpy(result, d_result, total_num * size, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(buffer);
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

  int n_samples = argc >= 4 ? atoi(argv[3]) : num_samples();
  int pdeg = argc >= 5 ? atoi(argv[4]) : BLOCK_SIZE;
  std::cout << "block size: " << pdeg << "\n";

  double iElaps;
  vector<vidType> initial = get_initial_transits(sample_size(-1) * n_samples, g.V());
  int step_count = sample_size(-1) * n_samples;
  int total_count = step_count;
  for (int step = 0; step < steps(); step++) {
    step_count *= sample_size(step);
    total_count += step_count;
  }
  vidType* result = new vidType[total_count];
  iElaps = multilayer_sample(g, initial, n_samples, total_count, result, pdeg);

  cout << "Time elapsed for sampling " << iElaps << " sec\n\n";
  delete[] result;

  return 0;
}