#include "graph.h"
#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cmath>
#include "khop_gpu.cuh"
#include "khop.h"
#include "graph_gpu_compressed.h"
#include "compressor.hh"
using namespace std;
using namespace cooperative_groups;

// const int BLOCK_DIM = 32;
// const vidType MAX_VIDTYPE = 0 - 1;
__global__ void khop_next0(GraphGPUCompressed g, vidType *result, int n_steps, int n_samples, int *step_counts, int total_threads, curandState *states) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id >= total_threads || thread_id >= n_samples) {
    return;
  }
  curandState local_state = states[thread_id];

  int step_count = step_counts[0];
  int t_begin = step_count * n_samples;
  int old_t_begin = 0;
  for (int step = 0; step < n_steps; step++) {
    int step_sample_size = step_counts[step + 1];
    int prev_step_count = step_count;
    step_count *= step_sample_size;
    for (int i = 0; i < step_count; i++) {
      int old_t_idx = old_t_begin + (thread_id * prev_step_count) + (i / step_sample_size);
      int t_idx = t_begin + (thread_id * step_count) + i;
      vidType old_t = result[old_t_idx];
      if (old_t == MAX_VIDTYPE) {
        result[t_idx] = MAX_VIDTYPE;
        continue;
      } 
      result[t_idx] = g.decode_cgr_naive<false>(old_t, local_state);
    }
    old_t_begin = t_begin;
    t_begin += step_count * n_samples;
  }
}

double multilayer_sample(Graph &g, vector<vidType>& initial, int n_samples, int total_num, int last_step_num, vidType* result, int block_size, bool use_uva) {
    int cur_num = initial.size();
    int n_steps = steps();
    vidType *d_result;
    int *step_counts = new int[n_steps + 1];
    int *d_step_counts;
    curandState *d_states;
    double alloc_t, rand_t, sample_t, dealloc_t;
    int v_size = sizeof(vidType);
    int e_size = sizeof(eidType);
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }

    int total_threads = 40000;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    sizes_list(n_steps, step_counts);
    alloc_t = seconds();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, total_num * v_size));
    CUDA_SAFE_CALL(cudaMemcpy(d_result, result, cur_num * v_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_step_counts, (steps() + 1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_step_counts, step_counts, (steps() + 1) * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_states, total_threads * sizeof(curandState)));
    alloc_t = seconds() - alloc_t;

    rand_t = seconds();
    setup_kernel<<<num_blocks,block_size>>>(d_states, total_threads);
    rand_t = seconds() - rand_t;
    std::cout << "Sampled random states in " << rand_t << " sec\n";

    GraphGPUCompressed gg(g, "cgr", g.get_degree_threshold(), 0, 1, use_uva);
    
    std::cout << "Starting sampling with " << total_threads << " threads...\n";
    sample_t = seconds();
    khop_next0<<<num_blocks,block_size>>>(gg, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    sample_t = seconds() - sample_t;
    std::cout << "Done sampling!" << std::endl;
  
    dealloc_t = seconds();
    CUDA_SAFE_CALL(cudaMemcpy(result, d_result, total_num * v_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_result));
    CUDA_SAFE_CALL(cudaFree(d_states));
    CUDA_SAFE_CALL(cudaFree(d_step_counts));
    dealloc_t = seconds() - dealloc_t;

    std::cout << "Time elapsed for allocating and copying " << alloc_t + dealloc_t << " sec\n\n";

    return sample_t;
}

int main(int argc, char* argv[]) {
  size_t memsize = print_device_info(0);
  Graph g;
  std::string in_prefix = argv[1];
  // std::string out_prefix = argv[2];
  std::string scheme = "cgr";
  bool permutated = false;
  bool compress_graph = false;
  int c;
  bool print = false;
  int n_samples = num_samples();
  int pdeg = BLOCK_SIZE;
  bool use_uva = false;
  while ((c = getopt(argc, argv, "cpn:d:u")) != -1) {
    switch (c) {
      case 'c':
        compress_graph = true;
        break;
      case 'p':
        print = true;
        break;
      case 'n':
        n_samples = atoi(optarg);
        break;
      case 'd':
        pdeg = atoi(optarg);
        break;
      case 'u':
        use_uva = true;
        break;
      default:
        abort();
    }
  }
  if (compress_graph) { 
    std::string out_prefix = argv[2];
    // save_compressed_graph_cgr(in_prefix, out_prefix); 
    return 0;
  }
  g.load_compressed_graph(in_prefix, scheme, permutated);
  // g.print_meta_data();
  std::cout << "LOADED COMPRESSED GRAPH\n" << std::endl;

  // int n_samples = argc >= 4 ? atoi(argv[3]) : num_samples();
  // int pdeg = argc >= 5 ? atoi(argv[4]) : BLOCK_SIZE;
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
  iElaps = multilayer_sample(g, initial, n_samples, total_count, step_count, result, pdeg, use_uva);

  std::cout << "Sampled total of " << total_count << " transits in " << steps() << " steps\n";
  std::cout << "Time elapsed for sampling " << iElaps << " sec\n\n";
  if (print) {
    std::cout << "results\n";
    int _size = sample_size(-1) * n_samples;
    int p_size = 0;
    for (int step = 0; step <= steps(); step++) {
      std::cout << "\n\n";
      for (int i = 0; i < _size; i++) {
        vidType res = result[i + p_size];
        std::cout << res << ", deg(" << g.get_degree_vbyte(res) << ")  |  ";
        // cout << i + p_size << " ";
      }
      p_size += _size;
      _size *= sample_size(step);
    }
  }
  std::cout << "\n";
  delete[] result;

  return 0;
}