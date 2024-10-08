#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include "../../include/deepwalk_gpu.cuh"
#include "../../include/graph.h"
#include "../../include/deepwalk.h"
#include "../../include/graph_gpu.h"
using namespace std;
using namespace cooperative_groups;

// const int BLOCK_DIM = 32;
// const vidType MAX_VIDTYPE = 0 - 1;
#define GAMMA 1

// Each sample is handled by a warp (32 threads)
__global__ void deepwalk_next_warp(GraphGPU g, int total_threads, int n_steps, int n_samples, int *step_counts, vidType *result, curandState *states) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = thread_id / WARP_SIZE;
    if (thread_id >= total_threads) {
      return;
    }
    curandState local_state = states[thread_id];
    
    int step_count = n_samples * step_counts[0];
    int t_begin = step_count;
    int old_t_begin = 0;
    grid_group grid = this_grid();
    for (int step = 0; step < n_steps; step++) {
      if (warp_id < step_count) {
        int step_sample_size = step_counts[step + 1];
        step_count *= step_sample_size;
        int old_t_idx = old_t_begin + warp_id;
        vidType old_t = result[old_t_idx];
        for (int i = thread_id % WARP_SIZE; i < step_sample_size; i += WARP_SIZE) {
          int t_idx = t_begin + (warp_id * step_sample_size) + i;
          if (old_t == MAX_VIDTYPE) {
            result[t_idx] = MAX_VIDTYPE;
          } else {
            vidType old_t_deg = g.get_degree(old_t);
            result[t_idx] = get_next_gpu(g, old_t, old_t_deg, local_state);
          }
        }
      } else {
        int step_sample_size = step_counts[step + 1];
        step_count *= step_sample_size;
      }
      old_t_begin = t_begin;
      t_begin += step_count;
      grid.sync();
    }
}

// Each thread is responsible for one node 
__global__ void deepwalk_next_tp(GraphGPU g, int total_threads, int n_steps, int n_samples, int *step_counts, vidType *result, curandState *states) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= total_threads) {
        return;
    }
    curandState local_state = states[thread_id];
    
    int step_count = n_samples * step_counts[0];
    int t_begin = step_count;
    int old_t_begin = 0;
    grid_group grid = this_grid();
    for (int step = 0; step < n_steps; step++) {
      int step_sample_size = step_counts[step + 1];
      step_count *= step_sample_size;
      for (int i = thread_id; i < step_count; i += total_threads) {
        int old_t_idx = old_t_begin + i / step_sample_size;
        vidType old_t = result[old_t_idx];
        int t_idx = t_begin + i;
        if (old_t == MAX_VIDTYPE) {
          result[t_idx] = MAX_VIDTYPE;
        } else {
          vidType old_t_deg = g.get_degree(old_t);
          result[t_idx] = get_next_gpu(g, old_t, old_t_deg, local_state);
        }
      }
      old_t_begin = t_begin;
      t_begin += step_count;
      grid.sync();
    }
}

// Each sample is handled by one thread (currently used in multilayer_sample)
__global__ void deepwalk_next_sp(GraphGPU g, int total_threads, int n_steps, int n_samples, int *step_counts, vidType *result, curandState *states) {
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
      vidType old_t = result[old_t_idx];
      int t_idx = t_begin + (thread_id * step_count) + i;
      if (old_t == MAX_VIDTYPE) {
        result[t_idx] = MAX_VIDTYPE;
      } else {
        vidType old_t_deg = g.get_degree(old_t);
        result[t_idx] = get_next_gpu(g, old_t, old_t_deg, local_state);
      }
    }
  old_t_begin = t_begin;
  t_begin += step_count * n_samples;
  }
}

// Host code - will call deepwalk_next_warp, deepwalk_next_tp, or deepwalk_next_sp
double multilayer_sample(Graph &g, vector<vidType>& initial, int n_samples, int total_num, int last_step_num, vidType* result, int block_size, bool use_uva) {
    GraphGPU gg(g, use_uva);
    g.deallocate();
    int cur_num = initial.size();
    int n_steps = steps();
    vidType *d_result;
    int *step_counts = new int[n_steps + 1];
    int *d_step_counts;
    curandState *d_states;
    double alloc_t, rand_t, sample_t, dealloc_t;
    int size = sizeof(vidType);
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }

    int max_threads = 80000;
    int threads_needed = last_step_num / sample_size(steps() - 1);

    int total_threads = max_threads;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    sizes_list(n_steps, step_counts);
    alloc_t = seconds();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, total_num * size));
    CUDA_SAFE_CALL(cudaMemcpy(d_result, result, cur_num * size, cudaMemcpyHostToDevice)); // ERROR?
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_step_counts, (steps() + 1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_step_counts, step_counts, (steps() + 1) * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_states, total_threads * sizeof(curandState)));
    alloc_t = seconds() - alloc_t;

    rand_t = seconds();
    setup_kernel<<<num_blocks, block_size>>>(d_states, total_threads);
    rand_t = seconds() - rand_t;
    std::cout << "Sampled random states in " << rand_t << " sec\n";

    std::cout << "Starting sampling with " << total_threads << " threads...\n";
    dim3 block(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);
    void *kernel_args[] = {&gg, &total_threads, &n_steps, &n_samples, &d_step_counts, &d_result, &d_states};
    sample_t = seconds();
    
    deepwalk_next_sp<<<num_blocks, block_size>>>(gg, total_threads, n_steps, n_samples, d_step_counts, d_result, d_states);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    sample_t = seconds() - sample_t;

    dealloc_t = seconds();
    CUDA_SAFE_CALL(cudaMemcpy(result, d_result, total_num * size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_result));
    CUDA_SAFE_CALL(cudaFree(d_states));
    CUDA_SAFE_CALL(cudaFree(d_step_counts));
    dealloc_t = seconds() - dealloc_t;

    std::cout << "Time elapsed for allocating and copying " << alloc_t + dealloc_t << " sec\n\n";

    return sample_t;
}

// This is where we call multilayer_sample
int main(int argc, char* argv[]) {
  size_t memsize = print_device_info(0);
  std::string in_prefix = argv[1];
  Graph g(in_prefix, 0, 0, 0, 0, 0);
  int c;
  bool print = false;
  int n_samples = num_samples(); // will be equal to g.V()
  int block_size = BLOCK_SIZE;
  bool use_uva = false;
  while ((c = getopt(argc, argv, "pn:d:ru")) != -1) {
    switch (c) {
      case 'p': //print results or not
        print = true;
        break;
      case 'n': //batch size
        n_samples = atoi(optarg);
        break;
      case 'd':
        block_size = atoi(optarg);
        break;
      case 'u': //if use unified virtual memory
        use_uva = true;
        break;
      default:
        abort();
    }
  }
  // g.print_meta_data();
  std::cout << "block size: " << block_size << "\n";

  double iElaps;
  n_samples = g.V() * GAMMA; // CHANGED FROM KHOP - GAMMA samples from each node
  vector<vidType> all_vertices = get_initial_transits(g.V(), g.V()); // will be a list of all vertices
  vector<vidType> initial = {}; // CHANGED FROM KHOP - include each node GAMMA times
  for (int i = 0; i < g.V(); i++) {
    for (int j = 0; j < GAMMA; j++) {
      initial.push_back(all_vertices[i]);
    }
  }
  int step_count = sample_size(-1) * n_samples; // CHANGED FROM KHOP - will be g.V() * GAMMA
  int total_count = step_count;
  for (int step = 0; step < steps(); step++) {
    step_count *= sample_size(step); // step_count will still be g.V() * GAMMA
    total_count += step_count; // in the end, total_count will be (steps() + 1) * g.V() * GAMMA
  }
  vidType* result = new vidType[total_count];
  iElaps = multilayer_sample(g, initial, n_samples, total_count, step_count, result, block_size, use_uva);
  cout << "Time elapsed for sampling " << total_count << " nodes: " << iElaps << " sec\n\n";
  if (print) {
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
  }
  cout << "\n";
  delete[] result;

  return 0;
}