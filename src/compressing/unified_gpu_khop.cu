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

template <int scheme = 0, bool delta = true, int pack_size = 4>
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
    }
  }
  __syncwarp();
  if (old_t_deg == 0) { return; }

  int next_warp_ptr = warp_start_ptr + sample_size;
  int round_threshold = WARP_SIZE;
  // int max_idx = random_idxs[next_warp_ptr - 1]; // only works for sorted random_idxs
  for (int i = thread_lane; i < old_t_deg; i += WARP_SIZE) {
    g.decode_vbyte_warp_thread<scheme,delta,pack_size>(old_t, adj_buffer, i);
    __syncwarp();
    int available_threads = min(WARP_SIZE, old_t_deg - (round_threshold - WARP_SIZE));
    for (int v_i = warp_start_ptr + thread_lane; v_i < warp_start_ptr + sample_size; v_i += available_threads) {
      int n_idx = random_idxs[v_i];
      if (n_idx >= round_threshold - WARP_SIZE && n_idx < round_threshold) {
        printf("n %d; r %d; v_i %d; w %d\n", n_idx, round_threshold, v_i, warp_start_ptr);
        int t_idx = t_begin + (warp_id * sample_size) + (v_i - warp_start_ptr);
        result[t_idx] = adj_buffer[n_idx % WARP_SIZE];
      }
    }
    round_threshold += WARP_SIZE;
    __syncwarp();
  }
}

template <int scheme = 0, bool delta = true, int pack_size = 4>
__global__ void khop_next2(GraphGPUCompressed g, vidType *result, int smem_bytes, int sample_size, int t_begin, int old_t_begin, int total_threads, curandState *states) {
  extern __shared__ int smem[];
  int *random_idxs = smem;
  int *round_idxs = smem + smem_bytes;
  __shared__ vidType adj_buffer[WARP_SIZE];
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id >= total_threads) {
    return;
  }
  int warp_id = thread_id / WARP_SIZE;
  int thread_lane = threadIdx.x % WARP_SIZE;
  int warp_lane = threadIdx.x / WARP_SIZE;
  int old_t_idx = old_t_begin + warp_id;
  vidType old_t = result[old_t_idx];
  vidType old_t_deg = 0;
  if (old_t != MAX_VIDTYPE) {
    old_t_deg = g.get_degree(old_t);
  }
  // sample fan out size num of random indices for single transit per warp
  curandState local_state = states[thread_id];
  for (int i = thread_lane; i < sample_size; i += WARP_SIZE) {
    if (old_t_deg == 0) { // no need to continue sampling indices for 0 degree vertices
      int t_idx = t_begin + (warp_id * sample_size) + i;
      result[t_idx] = MAX_VIDTYPE;
    }
    else {
      random_idxs[threadIdx.x] = (int)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
    }
  }
  __syncwarp();
  if (old_t_deg == 0) { return; }

  // sort each warp's random indices
  // from (threadIdx.x / WARP_SIZE * sample_size, (threadIdx.x / WARP_SIZE + 1) * sample_size)
  // __syncwarp();

  int warp_start_ptr = warp_lane * sample_size;
  for (int i = warp_start_ptr + thread_lane; i < warp_start_ptr + sample_size; i += WARP_SIZE) {
    if (i == 0) {
      round_idxs[i] = warp_start_ptr;
      continue;
    }
    // int ii = warp_start_ptr + i;
    int prev_round = random_idxs[i - 1] / WARP_SIZE;
    int curr_round = random_idxs[i] / WARP_SIZE;
    if (prev_round != curr_round) {
      round_idxs[i] = i;
    }
    else {
      round_idxs[i] = -1;
    }
  }

  int round = 1;
  int r_ii = warp_start_ptr;
  int next_warp_ptr = warp_start_ptr + sample_size;
  int max_idx = random_idxs[next_warp_ptr - 1];
  for (int i = thread_lane; i < max_idx; i += WARP_SIZE) {
    g.decode_vbyte_warp_thread<scheme,delta,pack_size>(old_t, adj_buffer);
    __syncwarp();
    while (round_idxs[r_ii] == -1) {
      r_ii++;
    }
    int round_idx = round_idxs[r_ii];
    int random_idx = random_idxs[round_idx];
    int round_threshold = round * WARP_SIZE;
    for (int v_i = thread_lane; v_i < sample_size; v_i += WARP_SIZE) {
      round_idx += v_i;
      if (round_idx >= next_warp_ptr) { break; }
      random_idx = random_idxs[round_idx];
      if (random_idx >= round_threshold) { break; }
      int t_idx = t_begin + (warp_id * sample_size) + (round_idx - warp_start_ptr);
      result[t_idx] = adj_buffer[random_idxs[round_idx] % WARP_SIZE];
      round_idx++;
      random_idx = random_idxs[round_idx];
    }
    round++;
    __syncwarp();
  }
}

__global__ void khop_next_with_buffer(GraphGPUCompressed g, vidType *result, int sample_size, int t_begin, int old_t_begin, vidType* buffer, int total_threads, curandState *states) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = thread_id / WARP_SIZE;
  if (thread_id >= total_threads) {
    return;
  }
  curandState local_state = states[thread_id];
  // curandState local_state;
  // curand_init(seed, thread_id, 0, &local_state);
  int old_t_idx = old_t_begin + warp_id;
  vidType old_t = result[old_t_idx];
  vidType old_t_deg = g.get_degree(old_t);
  vidType *adj = buffer + (g.get_max_degree() * warp_id);
  vidType new_t = MAX_VIDTYPE;
  for (int i = thread_id % WARP_SIZE; i < sample_size; i += WARP_SIZE) {
    int t_idx = t_begin + (warp_id * sample_size) + i;
    new_t = next_gpu2(adj, old_t_deg, local_state);
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

vidType* decompress_top_degrees(Graph &g, float percent) {
  int num = ceil(percent * g.V());
  vector<vidType> sizes = g.get_sizes_vbyte();
  vector<vidType> sorted_id = sort_by_sizes(sizes);
  int total_size = 0;
  for (int i = 0; i < num; i++) {
    vidType v = sorted_id[i];
    total_size += g.get_degree_vbyte(v);
  }
  vidType adj_list[total_size];
  vidType *output = adj_list;
  for (int i = 0; i < num; i++) {
    vidType deg = g.decode_vertex_vbyte(sorted_id[i], output, "streamvbyte");
    output += deg;
  }
  return adj_list;
}

// 40000 * 25 * 10 + 40000 * 25 + 40000
double multilayer_sample(Graph &g, vector<vidType>& initial, int n_samples, int total_num, vidType* result, int pdeg=128) {
    GraphGPUCompressed gg(g, "streamvbyte", g.get_degree_threshold(), 0, 1, true);
    int cur_num = initial.size();
    vidType *d_result;
    // vidType *d_result, *buffer;
    curandState *d_states;
    vidType max_degree = g.get_max_degree();
    int block_size = pdeg;
    Timer alloc_t, sample_t, dealloc_t;
    int size = sizeof(vidType);
    size_t warps_per_block = block_size / WARP_SIZE;
    size_t nblocks = (g.V()-1)/warps_per_block+1;
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }

    vidType *top_degrees = decompress_top_degrees(g, 0.1);
    vidType *d_top_degrees;
    int n_top = sizeof(top_degrees) / size;

    // std::cout << "Allocating buffer for decompressed adjacency lists\n";
    alloc_t.Start();
    // allocate_gpu_buffer(size_t(max_degree) * warps_per_block * nblocks, buffer);
    cudaMalloc((void **)&d_result, total_num * size);
    cudaMemcpy(d_result, result, cur_num * size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_top_degrees, n_top);
    cudaMemcpy(d_top_degrees, top_degrees, n_top, cudaMemcpyHostToDevice);

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
        // int num_blocks = (step_count + block_size - 1) / block_size;
        // assign_warps<<<num_blocks,block_size>>>(gg, d_result, step_sample_size, old_t_begin, buffer, total_threads);
        // cudaDeviceSynchronize();
        // khop_next_with_buffer<<<num_blocks,block_size>>>(gg, d_result, step_sample_size, t_begin, old_t_begin, buffer, total_threads, d_states);
        int shared_numbytes = step_sample_size * (block_size / WARP_SIZE) * sizeof(int);
        khop_next<<<num_blocks,block_size,shared_numbytes>>>(gg, d_result, step_sample_size, t_begin, old_t_begin, total_threads, d_states);
        // khop_next2<<<num_blocks,block_size,shared_numbytes*2>>>(gg, d_result, shared_numbytes, step_sample_size, t_begin, old_t_begin, total_threads, d_states);
        cudaDeviceSynchronize();
        old_t_begin += prev_step_count;
    }
    sample_t.Stop();

    dealloc_t.Start();
    cudaMemcpy(result, d_result, total_num * size, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    // cudaFree(buffer);
    cudaFree(d_states);
    cudaFree(d_top_degrees);
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