#include "graph.h"
#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include "khop_gpu.cuh"
#include "khop.h"
#include "graph_gpu.h"
using namespace std;
using namespace cooperative_groups;

// const int BLOCK_DIM = 32;
// const vidType MAX_VIDTYPE = 0 - 1;

__global__ void khop_next_relaunch(GraphGPU g, vidType *result, int sample_size, int t_begin, int old_t_begin, int total_threads, curandState *states) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = thread_id / WARP_SIZE;
  if (thread_id >= total_threads) {
    return;
  }
  curandState local_state = states[thread_id];
  int old_t_idx = old_t_begin + warp_id;
  vidType old_t = result[old_t_idx];
  for (int i = thread_id % WARP_SIZE; i < sample_size; i += WARP_SIZE) {
    int t_idx = t_begin + (warp_id * sample_size) + i;
    if (old_t == MAX_VIDTYPE) {
      result[t_idx] = MAX_VIDTYPE;
      return;
    }
    vidType old_t_deg = g.get_degree(old_t);
    result[t_idx] = get_next_gpu(g, old_t, old_t_deg, local_state);
  }
}

__global__ void khop_next(GraphGPU g, int total_threads, int n_steps, int *step_counts, vidType *result, curandState *states) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = thread_id / WARP_SIZE;
    if (thread_id >= total_threads) {
        return;
    }
    curandState local_state = states[thread_id];
    
    int step_count = step_counts[0];
    int t_begin = 0;
    int old_t_begin = 0;
    for (int step = 0; step < n_steps; step++) {
        int old_t_idx = old_t_begin + warp_id;
        vidType old_t = result[old_t_idx];
        int step_sample_size = step_counts[step + 1];
        if (thread_id >= step_count * WARP_SIZE) {
            continue;
        }
        old_t_begin += step_count;
        step_count *= step_sample_size;
        for (int i = thread_id % WARP_SIZE; i < step_sample_size; i += WARP_SIZE) {
            int t_idx = t_begin + (warp_id * step_sample_size) + i;
            if (old_t == MAX_VIDTYPE) {
                result[t_idx] = MAX_VIDTYPE;
                return;
            }
            vidType old_t_deg = g.get_degree(old_t);
            result[t_idx] = get_next_gpu(g, old_t, old_t_deg, local_state);
        }
        __syncthreads();
    }
}

// 40000 * 25 * 10 + 40000 * 25 + 40000
double multilayer_sample_relaunch(Graph &g, vector<vidType>& initial, int n_samples, int total_num, vidType* result, int block_size, bool use_uva) {
    GraphGPU gg(g, use_uva);
    int cur_num = initial.size();
    vidType *d_result;
    curandState *d_states;
    Timer alloc_t, sample_t, dealloc_t;
    int size = sizeof(vidType);
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }

    alloc_t.Start();
    cudaMalloc((void **)&d_result, total_num * size);
    cudaMemcpy(d_result, result, cur_num * size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&d_states, total_num * sizeof(curandState));
    alloc_t.Stop();

    std::cout << "Sampling random states\n";
    int total_num_blocks = (total_num + block_size - 1) / block_size;
    setup_kernel<<<total_num_blocks,block_size>>>(d_states);

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
        khop_next_relaunch<<<num_blocks,block_size>>>(gg, d_result, step_sample_size, t_begin, old_t_begin, total_threads, d_states);
        cudaDeviceSynchronize();
        old_t_begin += prev_step_count;
    }
    sample_t.Stop();

    dealloc_t.Start();
    cudaMemcpy(result, d_result, total_num * size, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(d_states);
    dealloc_t.Stop();

    std::cout << "Time elapsed for allocating and copying " << alloc_t.Seconds() + dealloc_t.Seconds() << " sec\n\n";

    return sample_t.Seconds();
}

double multilayer_sample(Graph &g, vector<vidType>& initial, int n_samples, int total_num, vidType* result, int block_size, bool use_uva) {
    GraphGPU gg(g, use_uva);
    int cur_num = initial.size();
    vidType *d_result;
    int *step_counts = new int[steps() + 1];
    int *d_step_counts;
    curandState *d_states;
    Timer alloc_t, sample_t, dealloc_t;
    int size = sizeof(vidType);
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }
    sizes_list(steps(), step_counts);
    step_counts[0] *= n_samples;
    alloc_t.Start();
    cudaMalloc((void **)&d_result, total_num * size);
    cudaMemcpy(d_result, result, cur_num * size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_step_counts, (steps() + 1) * sizeof(int));
    cudaMemcpy(d_step_counts, step_counts, (steps() + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_states, total_num * sizeof(curandState));
    alloc_t.Stop();

    std::cout << "Sampling random states\n";
    int total_num_blocks = (total_num + block_size - 1) / block_size;
    setup_kernel<<<total_num_blocks,block_size>>>(d_states);

    std::cout << "Starting sampling with " << block_size << " threads...\n";
    sample_t.Start();
    int total_threads = total_num * WARP_SIZE;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    khop_next<<<num_blocks,block_size>>>(gg, total_threads, steps(), d_step_counts, d_result, d_states);
    cudaDeviceSynchronize();
    sample_t.Stop();

    dealloc_t.Start();
    cudaMemcpy(result, d_result, total_num * size, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(d_states);
    cudaFree(d_step_counts);
    dealloc_t.Stop();

    std::cout << "Time elapsed for allocating and copying " << alloc_t.Seconds() + dealloc_t.Seconds() << " sec\n\n";

    return sample_t.Seconds();
}


int main(int argc, char* argv[]) {
  size_t memsize = print_device_info(0);
  std::string in_prefix = argv[1];
  Graph g(in_prefix, 0, 0, 0, 0, 0);
  int c;
  bool print = false;
  int n_samples = num_samples();
  int block_size = BLOCK_SIZE;
  int r = false;
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
      case 'r': //if use the original khop that relaunches every step
        r = true;
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
  vector<vidType> initial = get_initial_transits(sample_size(-1) * n_samples, g.V());
  int step_count = sample_size(-1) * n_samples;
  int total_count = step_count;
  for (int step = 0; step < steps(); step++) {
    step_count *= sample_size(step);
    total_count += step_count;
  }
  vidType* result = new vidType[total_count];
  if (r) {
    iElaps = multilayer_sample_relaunch(g, initial, n_samples, total_count, result, block_size, use_uva);
  } else {
    iElaps = multilayer_sample(g, initial, n_samples, total_count, result, block_size, use_uva);
  }
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