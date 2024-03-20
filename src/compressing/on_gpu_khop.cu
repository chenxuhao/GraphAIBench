#include "graph.h"
#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include "khop_gpu.cuh"
#include "khop.h"
#include "graph_gpu_compressed.h"
#include "compressor.hh"
using namespace std;
using namespace cooperative_groups;

// const int BLOCK_DIM = 32;
// const vidType MAX_VIDTYPE = 0 - 1;

__global__ void khop_next(GraphGPUCompressed g, vidType *result, int sample_size, int t_begin, int old_t_begin, vidType* buffer, int total_threads, int seed) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = thread_id / WARP_SIZE;
  if (thread_id >= total_threads) {
    return;
  }
  curandState ix_state;
  curand_init(seed, thread_id, 0, &ix_state);
  int old_t_idx = old_t_begin + warp_id;
  vidType old_t = result[old_t_idx];
  vidType old_t_deg = g.get_degree(old_t);
  vidType *adj = buffer + (g.get_max_degree() * warp_id);
  vidType new_t = MAX_VIDTYPE;
  for (int i = thread_id % WARP_SIZE; i < sample_size; i += WARP_SIZE) {
    int t_idx = t_begin + (warp_id * sample_size) + i;
    new_t = next_gpu2(adj, old_t_deg, ix_state);
    result[t_idx] = new_t;
  }
}

// template <int scheme = 0, bool delta = true, int pack_size = 4>
__global__ void assign_warps(GraphGPUCompressed g, vidType *result, int sample_size, int old_t_begin, vidType* buffer, int total_threads) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = thread_id / WARP_SIZE;
  if (thread_id >= total_threads) {
    return;
  }
  int old_t_idx = old_t_begin + warp_id;
  vidType old_t = result[old_t_idx];
  vidType old_t_deg = g.get_degree(old_t);
  vidType *adj = buffer + (g.get_max_degree() * warp_id);
  for (int i = thread_id % WARP_SIZE; i < old_t_deg; i += WARP_SIZE) {
    auto deg_v = decompress_edge(g, old_t, warp_id, adj);
  }
}


__global__ void khop_sample(GraphGPUCompressed g, vidType* result, int sample_size, int t_begin, int old_t_begin, int total_threads, vidType *buffer, int seed) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= total_threads) {
        return;
    }
    curandState ix_state;
    curand_init(seed, thread_id, 0, &ix_state);
    int t_idx = t_begin + thread_id;
    int old_t_idx = old_t_begin + thread_id / sample_size;
    vidType old_t = result[old_t_idx];
    if (old_t == MAX_VIDTYPE) {
        result[t_idx] = MAX_VIDTYPE;
    }
    else {
        vidType new_t = next_gpu(g, old_t, thread_id, buffer, g.get_max_degree(), ix_state);
        printf("old_t: %d, new_t: %d\n", old_t, new_t);
        result[t_idx] = new_t;
    }
} 

// 40000 * 25 * 10 + 40000 * 25 + 40000
double multilayer_sample(Graph &g, vector<vidType>& initial, int n_samples, int total_num, vidType* result, int pdeg=128, int seed=0) {
    GraphGPUCompressed gg(g, "streamvbyte", g.get_degree_threshold());
    int cur_num = initial.size();
    vidType *d_result, *buffer;
    vidType max_degree = g.get_max_degree();
    int block_size = pdeg;
    Timer alloc_t, sample_t, dealloc_t;
    int size = sizeof(vidType);
    size_t warps_per_block = block_size / WARP_SIZE;
    size_t nblocks = (g.V()-1)/warps_per_block+1;
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }

    std::cout << "Allocating buffer for decompressed adjacency lists\n";
    std::cout << "size_t(max_degree) " << size_t(max_degree) << " total " << size_t(max_degree) * warps_per_block * nblocks << "\n";
    alloc_t.Start();
    allocate_gpu_buffer(size_t(max_degree) * warps_per_block * nblocks, buffer);
    // allocate_gpu_buffer(size_t(g.get_max_degree()) * warps_per_block * nblocks, buffer);
    cudaMalloc((void **)&d_result, total_num * size);
    cudaMemcpy(d_result, result, cur_num * size, cudaMemcpyHostToDevice);
    alloc_t.Stop();

    std::cout << "Starting sampling with " << block_size << " threads...\n";
    sample_t.Start();
    int step_count = sample_size(-1) * n_samples;
    int prev_step_count = n_samples;
    int t_begin = 0;
    int old_t_begin = 0;
    for (int step = 0; step < steps(); step++) {
        std::cout << "STEP " << step << "\n";
        t_begin += step_count;
        int step_sample_size = sample_size(step);
        step_count *= step_sample_size;
        prev_step_count *= sample_size(step-1);
        int total_threads = prev_step_count * WARP_SIZE;
        int num_blocks = (total_threads + block_size - 1) / block_size;
        // int num_blocks = (step_count + block_size - 1) / block_size;
        assign_warps<<<num_blocks,block_size>>>(gg, d_result, step_sample_size, old_t_begin, buffer, total_threads);
        // khop_sample<<<num_blocks,block_size>>>(gg, d_result, step_sample_size, t_begin, old_t_begin, step_count, buffer, seed);
        cudaDeviceSynchronize();
        khop_next<<<num_blocks,block_size>>>(gg, d_result, step_sample_size, t_begin, old_t_begin, buffer, total_threads, seed);
        cudaDeviceSynchronize();
        old_t_begin += prev_step_count;
    }
    sample_t.Stop();

    dealloc_t.Start();
    cudaMemcpy(result, d_result, total_num * size, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(buffer);
    dealloc_t.Stop();

    std::cout << "Time elapsed for allocating and copying " << alloc_t.Seconds() + dealloc_t.Seconds() << " sec\n\n";

    return sample_t.Seconds();
}


int main(int argc, char* argv[]) {
  size_t memsize = print_device_info(0);
  Graph g;
  std::string in_prefix = argv[1];
  std::string out_prefix = argv[2];
  std::string scheme = "streamvbyte";
  bool permutated = false;
//   save_compressed_graph(in_prefix, out_prefix);
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
  std::cout << "results\n";
  int _size = sample_size(-1) * n_samples;
  int p_size = 0;
  for (int step = 0; step <= steps(); step++) {
    std::cout << "\n";
    for (int i = 0; i < _size; i++) {
        std::cout << result[i + p_size] << " ";
        // cout << i + p_size << " ";
    }
    p_size += _size;
    _size *= sample_size(step);
  }
  cout << "\n";
  delete[] result;

  return 0;
}