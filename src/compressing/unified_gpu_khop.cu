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

// __global__ void khop_next0(GraphGPUCompressed g, vidType *result, int n_steps, int n_samples, int *step_counts, int total_threads, curandState *states) {
__global__ void khop_next0(GraphGPUCompressed low_g, GraphGPU high_g, vidType first_low, vidType *result, int n_steps, int n_samples, int *step_counts, int total_threads, curandState *states) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id >= total_threads || thread_id >= n_samples) {
    return;
  }
  curandState local_state = states[thread_id];

  vidType high_deg = high_g.V();
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
      result[t_idx] = 0;
      if (old_t == MAX_VIDTYPE) {
        result[t_idx] = MAX_VIDTYPE;
      } 
      else {
        if (old_t < high_deg) {
          vidType old_t_deg = high_g.get_degree(old_t);
          if (old_t_deg == 0) {
            result[t_idx] = MAX_VIDTYPE;
            continue;
          }
          eidType n_idx = (eidType)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
          result[t_idx] = high_g.N(old_t, n_idx);
          // printf("HIGH deg %d; n_idx %d; t_idx %d; old_t %d; t %d\n", old_t_deg, (int)n_idx, t_idx, result[old_t_idx], result[t_idx]);
        } 
        else if (old_t >= first_low) {
          old_t -= first_low;
          vidType old_t_deg = low_g.get_degree(old_t);
          if (old_t_deg == 0) {
            result[t_idx] = MAX_VIDTYPE;
            continue;
          }
          eidType n_idx = (eidType)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
          result[t_idx] = low_g.decode_vbyte_sums(old_t, n_idx);
          // printf("LOW deg %d; n_idx %d; t_idx %d; old_t %d; t %d\n", old_t_deg, (int)n_idx, t_idx, result[old_t_idx], result[t_idx]);
        }
        else {
          // printf("MED t_idx %d; old_t %d; t %d\n", t_idx, result[old_t_idx], 0);
          result[t_idx] = 0; // placeholder
        }
      }
    }
    old_t_begin = t_begin;
    t_begin += step_count * n_samples;
    // printf("\n");
  }
}

__global__ void khop_next(GraphGPUCompressed g, vidType *result, int sample_size, int t_begin, int old_t_begin, int total_threads, curandState *states) {
  extern __shared__ int random_idxs[];
  __shared__ vidType adj_buffer[WARP_SIZE];
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id >= total_threads) {
    return;
  }
  int warp_id = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x % WARP_SIZE;
  int warp_lane = threadIdx.x / WARP_SIZE;
  int warp_start_ptr = warp_lane * sample_size;
  int old_t_idx = old_t_begin + warp_id;
  vidType old_t = result[old_t_idx];
  vidType old_t_deg = 0;
  if (old_t != MAX_VIDTYPE) {
    old_t_deg = g.get_degree(old_t);
  }
  // sample fan out size num of random indices for single transit per warp
  curandState local_state = states[thread_id];
  for (int i = warp_start_ptr + thread_lane; i < warp_start_ptr + sample_size; i += WARP_SIZE) {
    if (old_t_deg == 0) { // no need to continue sampling indices for 0 degree vertices
      int t_idx = t_begin + (warp_id * sample_size) + (i - warp_start_ptr);
      result[t_idx] = MAX_VIDTYPE;
    }
    else {
      random_idxs[i] = (int)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
      // printf("old %d; r[i] %d\n", old_t, random_idxs[i]);
    }
  }
  __syncwarp();
  if (old_t_deg == 0) { return; }

  for (int i = thread_lane; i < sample_size; i += WARP_SIZE) {
    int r_i = warp_start_ptr + i;
    int n_idx = random_idxs[r_i];
    g.decode_vbyte_sums(old_t, adj_buffer, n_idx);
    int t_idx = t_begin + (warp_id * sample_size) + i;
    result[t_idx] = adj_buffer[thread_lane];
  }
}

__global__ void khop_next1(GraphGPUCompressed g, vidType *result, int sample_size, int t_begin, int old_t_begin, int total_threads, curandState *states) {
  extern __shared__ int random_idxs[];
  __shared__ vidType adj_buffer[WARP_SIZE];
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id >= total_threads) {
    return;
  }
  int warp_id = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x % WARP_SIZE;
  int warp_lane = threadIdx.x / WARP_SIZE;
  int warp_start_ptr = warp_lane * sample_size;
  int old_t_idx = old_t_begin + warp_id;
  vidType old_t = result[old_t_idx];
  vidType old_t_deg = 0;
  if (old_t != MAX_VIDTYPE) {
    old_t_deg = g.get_degree(old_t);
  }
  // sample fan out size num of random indices for single transit per warp
  curandState local_state = states[thread_id];
  for (int i = warp_start_ptr + thread_lane; i < warp_start_ptr + sample_size; i += WARP_SIZE) {
    if (old_t_deg == 0) { // no need to continue sampling indices for 0 degree vertices
      int t_idx = t_begin + (warp_id * sample_size) + (i - warp_start_ptr);
      result[t_idx] = MAX_VIDTYPE;
    }
    else {
      random_idxs[i] = (int)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
    }
  }
  __syncwarp();
  if (old_t_deg == 0) { return; }

  int next_warp_ptr = warp_start_ptr + sample_size;
  int round_threshold = WARP_SIZE;
  // int max_idx = random_idxs[next_warp_ptr - 1]; // only works for sorted random_idxs
  for (int i = thread_lane; i < old_t_deg; i += WARP_SIZE) {
    g.decode_vbyte_thread(old_t, adj_buffer, i);
    __syncwarp();
    int available_threads = min(WARP_SIZE, old_t_deg - (round_threshold - WARP_SIZE));
    for (int v_i = warp_start_ptr + thread_lane; v_i < warp_start_ptr + sample_size; v_i += available_threads) {
      int n_idx = random_idxs[v_i];
      if (n_idx >= round_threshold - WARP_SIZE && n_idx < round_threshold) {
        // printf("n %d; r %d; v_i %d; w %d\n", n_idx, round_threshold, v_i, warp_start_ptr);
        int t_idx = t_begin + (warp_id * sample_size) + (v_i - warp_start_ptr);
        result[t_idx] = adj_buffer[n_idx % WARP_SIZE];
      }
    }
    round_threshold += WARP_SIZE;
    __syncwarp();
  }
}

// threshold should be divisible by warp size (32)
void decompress_top_degrees(Graph &g, vidType *edges, eidType *vertices, int last_v) {
  vertices[0] = 0;
  for (int i = 0; i < last_v; i++) {
    vidType deg = g.decode_vertex_vbyte(i, edges, "streamvbyte");
    edges += deg;
    vertices[i+1] = deg;
  }
}

double multilayer_sample(Graph &g, vector<vidType>& initial, int n_samples, int total_num, int last_step_num, vidType* result, int block_size, int h_deg) {
    // GraphGPUCompressed gg(g, "streamvbyte", g.get_degree_threshold(), 0, 1, true);
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

    vidType last_high = 0;
    while (g.get_degree_vbyte(last_high) > h_deg) {
      last_high++;
    }
    eidType total_deg = 0;
    for (int i = 0; i < last_high; i++) {
      total_deg += g.get_degree_vbyte(i);
    }
    GraphGPU high_subg(g, last_high, total_deg);

    int l_deg = WARP_SIZE;
    vidType last_med = g.V() - 1;
    while (g.get_degree_vbyte(last_med) <= l_deg) {
      last_med--;
    }
    vidType first_low = last_med + 1;
    auto g_rptr = g.rowptr_compressed();
    total_deg = g_rptr[g.V()] - g_rptr[first_low];
    GraphGPUCompressed low_subg(g, first_low, g.V() - first_low, total_deg);
    std::cout << "last_med " << last_med << " deg " << g.get_degree_vbyte(last_med) << " first_low " << first_low << " deg " << g.get_degree_vbyte(first_low) << std::endl;
    eidType ttt = 0;
    auto g_c = g.colidx_compressed();
    for (int i = first_low; i < g.V(); i++) {
      ttt += g_c[g_rptr[i]];
    }
    std::cout << "total edges in low subgraph " << ttt << " and " << total_deg << std::endl;

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

    std::cout << "Starting sampling with " << total_threads << " threads...\n";
    // dim3 block(block_size, 1, 1);
    // dim3 grid(num_blocks, 1, 1);    
    // void *kernel_args[] = {&gg, &d_result, &n_steps, &n_samples, &d_step_counts, &total_threads, &d_states};
    // void *kernel_args[] = {&low_subg, &high_subg, &d_result, &n_steps, &d_step_counts, &total_threads, &d_states};
    sample_t = seconds();
    khop_next0<<<num_blocks,block_size>>>(low_subg, high_subg, first_low, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
    // khop_next0<<<num_blocks,block_size>>>(gg, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
    // cudaLaunchCooperativeKernel((void*)(khop_next0), grid, block, kernel_args);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    sample_t = seconds() - sample_t;

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
  std::string out_prefix = argv[2];
  std::string scheme = "streamvbyte";
  bool permutated = false;
  bool compress_graph = false;
  int c;
  bool print = false;
  int n_samples = num_samples();
  int pdeg = BLOCK_SIZE;
  int high_deg = 256;
  while ((c = getopt(argc, argv, "cpn:d:h:")) != -1) {
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
      case 'h':
        high_deg = atoi(optarg);
        break;
      default:
        abort();
    }
  }
  if (compress_graph) { save_compressed_graph(in_prefix, out_prefix); }
  g.load_compressed_graph(out_prefix, scheme, permutated);
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
  iElaps = multilayer_sample(g, initial, n_samples, total_count, step_count, result, pdeg, high_deg);

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