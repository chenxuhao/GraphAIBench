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
#include <fstream>
#include <string>
using namespace std;
using namespace cooperative_groups;

// const int BLOCK_DIM = 32;
// const vidType MAX_VIDTYPE = 0 - 1;

template <int scheme = 0, bool delta = true, int pack_size = 4>
__global__ void khop_next(GraphGPUCompressed g, int buffer_offset, vidType *result, int n_steps, int n_samples, int *step_counts, int total_threads, curandState *states) {
    extern __shared__ vidType smem[];
    vidType *buffer = smem;
    int *random_idxs = (int*)&buffer[buffer_offset];
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= total_threads) {
        return;
    }
    int warp_id = thread_id / WARP_SIZE;
    curandState local_state = states[thread_id];
    int thread_lane = threadIdx.x % WARP_SIZE;
    int warp_lane = threadIdx.x / WARP_SIZE;
    int step_count = step_counts[0];
    int t_begin = step_count * n_samples;
    int old_t_begin = 0;
    for (int step = 0; step < n_steps; step++) {
      int step_sample_size = step_counts[step + 1];
      int prev_step_count = step_count;
      step_count *= step_sample_size;
      int warp_start_ptr = warp_lane * step_count;
      int *warp_idxs = random_idxs + warp_start_ptr;
      
      for (int i = thread_lane; i < step_count; i += WARP_SIZE) {
          int old_t_idx = old_t_begin + (warp_id * prev_step_count) + (i / step_sample_size);
          vidType old_t = result[old_t_idx];
          vidType old_t_deg = g.get_degree(old_t);
          warp_idxs[i] = (vidType)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
      }
      __syncwarp();

      for (int j = 0; j < prev_step_count; j++) {
        vidType prefix = 0;
        vidType bytes_prefix = 0;
        int old_t_idx = old_t_begin + (warp_id * prev_step_count) + j;
        int max_n_i = 0;
        for (int ni = 0; ni < step_sample_size; ni++) {
          int curr_n = warp_idxs[(j * step_sample_size) + ni];
          if (curr_n > max_n_i) max_n_i = curr_n;
        }
        vidType old_t = result[old_t_idx];
        vidType *adj_buffer = buffer + warp_lane * WARP_SIZE;
        int round_threshold = WARP_SIZE;
        int last_check = WARP_SIZE - (max_n_i % WARP_SIZE) + max_n_i;
        if (max_n_i == 0) last_check = 0;
        for (int i = thread_lane; i < last_check; i += WARP_SIZE) {
            bytes_prefix = g.decode_1warp<scheme,delta,pack_size>(old_t, adj_buffer, prefix, bytes_prefix, i);
            __syncwarp();
            prefix = adj_buffer[31];
            for (int v_i = thread_lane; v_i < step_sample_size; v_i += WARP_SIZE) {
                int n_idx = warp_idxs[(j * step_sample_size) + v_i];
                if (n_idx >= round_threshold - WARP_SIZE && n_idx < round_threshold) {
                    int t_idx = t_begin + (warp_id * step_count) + (j * step_sample_size) + v_i;
                    result[t_idx] = adj_buffer[n_idx % WARP_SIZE];
                    // printf("pref: %d; old_t: %d; n_idx: %d; deg: %d; t: %d\n", prefix, old_t, n_idx, max_n_i, result[t_idx]);
                }
            }
            round_threshold += WARP_SIZE;
            __syncwarp();
        }
      }
      old_t_begin = t_begin;
      t_begin += step_count * n_samples;
      __syncthreads();
    }
}

template <int scheme = 0, bool delta = true, int pack_size = 4>
__global__ void khop_next_3subs(GraphGPUCompressed low_g, GraphGPUCompressed med_g, GraphGPU high_g, int buffer_offset, vidType low_deg, vidType first_low, vidType interval, vidType *result, int n_steps, int n_samples, int *step_counts, int total_threads, curandState *states) {
    extern __shared__ vidType smem[];
    vidType *buffer = smem;
    float *random_idxs = (float*)&buffer[buffer_offset];
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= total_threads) {
        return;
    }

    curandState local_state = states[thread_id];
    int thread_lane = threadIdx.x % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int warp_lane = threadIdx.x / WARP_SIZE;
    vidType first_med = high_g.V();

    int step_count = step_counts[0];
    int t_begin = step_count * n_samples;
    int old_t_begin = 0;
    for (int step = 0; step < n_steps; step++) {
      int step_sample_size = step_counts[step + 1];
      int prev_step_count = step_count;
      step_count *= step_sample_size;
      int warp_start_ptr = warp_lane * step_count;
      float *warp_idxs = random_idxs + warp_start_ptr;

      for (int i = thread_lane; i < step_count; i += WARP_SIZE) {
          int old_t_idx = old_t_begin + (warp_id * prev_step_count) + (i / step_sample_size);
          vidType old_t = result[old_t_idx];
          // vidType old_t_deg = g.get_degree(old_t);
          warp_idxs[i] = curand_uniform(&local_state);
      }
      __syncwarp();

      for (int j = 0; j < prev_step_count; j++) {
        int old_t_idx = old_t_begin + (warp_id * prev_step_count) + j;
        vidType old_t = result[old_t_idx];
        int t_idx = t_begin + (warp_id * step_count) + (j * step_sample_size);
        vidType *adj_buffer = buffer + warp_lane * WARP_SIZE;
        if (old_t == MAX_VIDTYPE) {
          result[t_idx] = MAX_VIDTYPE;
        }
        else {
          if (old_t < first_med) {
            for (int ni = thread_lane; ni < step_sample_size; ni+=WARP_SIZE) {
              vidType old_t_deg = high_g.get_degree(old_t);
              int n = (int)(ceil(warp_idxs[(j * step_sample_size) + ni] * old_t_deg) - 1);
              // int n = (int)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
              // printf("HIGH n_idx %d; t_idx %d; old_t %d\n", n, t_idx+ni, result[old_t_idx]);
              result[t_idx+ni] = high_g.N(old_t, n);
              // printf("HIGH n_idx %d; t_idx %d; old_t %d; t %d\n", n, t_idx+ni, result[old_t_idx], result[t_idx+ni]);
            }
          } else if (old_t >= first_low) {
            old_t -= first_low;
            vidType old_t_deg = low_g.get_degree(old_t);
            vidType prefix = 0;
            vidType bytes_prefix = 0;
            int last_check = (low_deg % WARP_SIZE == 0)? low_deg: WARP_SIZE - (low_deg % WARP_SIZE) + low_deg;
            for (int r = thread_lane; r < last_check; r += WARP_SIZE) {
              int round_lower = (r / WARP_SIZE) * WARP_SIZE;
              bytes_prefix = low_g.decode_1warp<scheme,delta,pack_size>(old_t, adj_buffer, prefix, bytes_prefix, r);
              __syncwarp();
              prefix = adj_buffer[31];
              for (int ni = thread_lane; ni < step_sample_size; ni+=WARP_SIZE) {
                if (old_t_deg == 0) {
                  result[t_idx+ni] = MAX_VIDTYPE;
                } else {
                  int n = (int)(ceil(warp_idxs[(j * step_sample_size) + ni] * old_t_deg) - 1);
                  // int n = (int)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
                  // printf("LOW n_idx %d; t_idx %d; old_t %d\n", n, t_idx+ni, result[old_t_idx]);
                  if (n >= round_lower && n < round_lower + WARP_SIZE) {
                    result[t_idx+ni] = adj_buffer[n % WARP_SIZE];
                    // printf("LOW round %d; pref %d; n_idx %d; t_idx %d; old_t %d; t %d\n", r, prefix, n, t_idx+ni, result[old_t_idx], result[t_idx+ni]);
                  }
                }
              }
              __syncwarp();
            }
          } else {
            old_t -= first_med;
            vidType old_t_deg = med_g.get_degree(old_t);
            for (int ni = thread_lane; ni < step_sample_size; ni+=WARP_SIZE) {
              int n = (int)(ceil(warp_idxs[(j * step_sample_size) + ni] * old_t_deg) - 1);
              // int n = (int)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
              // printf("MED deg %d; n_idx %d; t_idx %d; old_t %d\n", old_t_deg, n, t_idx+ni, result[old_t_idx]);
              result[t_idx+ni] = med_g.decode_vbyte_prefix(old_t, n, interval);
              // printf("MED deg %d; n_idx %d; t_idx %d; old_t %d; t %d\n", old_t_deg, n, t_idx+ni, result[old_t_idx], result[t_idx+ni]);
            }
          }
        }
        __syncwarp();
      }
      old_t_begin = t_begin;
      t_begin += step_count * n_samples;
      __syncthreads();
    }
}

template <int scheme = 0, bool delta = true, int pack_size = 4>
__global__ void khop_next_4subs(GraphGPUCompressed low_g, GraphGPUCompressed med_g, GraphGPUCompressed high_g, GraphGPU uncomp_g, int buffer_offset, vidType low_deg, vidType first_low, vidType first_med, vidType first_high, vidType interval, vidType *result, int n_steps, int n_samples, int *step_counts, int total_threads, curandState *states) {
    extern __shared__ vidType smem[];
    vidType *buffer = smem;
    float *random_idxs = (float*)&buffer[buffer_offset];
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= total_threads) {
        return;
    }

    curandState local_state = states[thread_id];
    int thread_lane = threadIdx.x % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int warp_lane = threadIdx.x / WARP_SIZE;

    int step_count = step_counts[0];
    int t_begin = step_count * n_samples;
    int old_t_begin = 0;
    for (int step = 0; step < n_steps; step++) {
      int step_sample_size = step_counts[step + 1];
      int prev_step_count = step_count;
      step_count *= step_sample_size;
      int warp_start_ptr = warp_lane * step_count;
      float *warp_idxs = random_idxs + warp_start_ptr;

      for (int i = thread_lane; i < step_count; i += WARP_SIZE) {
          int old_t_idx = old_t_begin + (warp_id * prev_step_count) + (i / step_sample_size);
          vidType old_t = result[old_t_idx];
          // vidType old_t_deg = g.get_degree(old_t);
          warp_idxs[i] = curand_uniform(&local_state);
      }
      __syncwarp();

      for (int j = 0; j < prev_step_count; j++) {
        int old_t_idx = old_t_begin + (warp_id * prev_step_count) + j;
        vidType old_t = result[old_t_idx];
        int t_idx = t_begin + (warp_id * step_count) + (j * step_sample_size);
        vidType *adj_buffer = buffer + warp_lane * WARP_SIZE;
        // printf("old t %d\n", old_t);
	if (old_t == MAX_VIDTYPE) {
          result[t_idx] = MAX_VIDTYPE;
        }
        else {
          if (old_t < first_high) {
            for (int ni = thread_lane; ni < step_sample_size; ni+=WARP_SIZE) {
              vidType old_t_deg = uncomp_g.get_degree(old_t);
              int n = (int)(ceil(warp_idxs[(j * step_sample_size) + ni] * old_t_deg) - 1);
              // int n = (int)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
              // printf("UNCOMP n_idx %d; t_idx %d; old_t %d\n", n, t_idx+ni, result[old_t_idx]);
              result[t_idx+ni] = uncomp_g.N(old_t, n);
              // printf("UNCOMP n_idx %d; old_deg %d; old_t %d; t %d\n", n, old_t_deg, result[old_t_idx], result[t_idx+ni]);
            }
          }
          else if (old_t < first_med) {
            old_t -= first_high;
            vidType old_t_deg = high_g.get_degree(old_t);
            for (int ni = thread_lane; ni < step_sample_size; ni+=WARP_SIZE) {
              int n = (int)(ceil(warp_idxs[(j * step_sample_size) + ni] * old_t_deg) - 1);
              result[t_idx+ni] = high_g.decode_vbyte_prefix(old_t, n, interval);
              // printf("HIGH n_idx %d; t_idx %d; old_t %d; t %d\n", n, t_idx+ni, result[old_t_idx], result[t_idx+ni]);
            }
          } else if (old_t >= first_low) {
            old_t -= first_low;
            vidType old_t_deg = low_g.get_degree(old_t);
            vidType prefix = 0;
            vidType bytes_prefix = 0;
            int last_check = (low_deg % WARP_SIZE == 0)? low_deg: WARP_SIZE - (low_deg % WARP_SIZE) + low_deg;
            for (int r = thread_lane; r < last_check; r += WARP_SIZE) {
              int round_lower = (r / WARP_SIZE) * WARP_SIZE;
              bytes_prefix = low_g.decode_1warp<scheme,delta,pack_size>(old_t, adj_buffer, prefix, bytes_prefix, r);
              __syncwarp();
              prefix = adj_buffer[31];
              for (int ni = thread_lane; ni < step_sample_size; ni+=WARP_SIZE) {
                if (old_t_deg == 0) {
                  result[t_idx+ni] = MAX_VIDTYPE;
                } else {
                  int n = (int)(ceil(warp_idxs[(j * step_sample_size) + ni] * old_t_deg) - 1);
                  // int n = (int)(ceil(curand_uniform(&local_state) * old_t_deg) - 1);
                  // printf("LOW n_idx %d; t_idx %d; old_t %d\n", n, t_idx+ni, result[old_t_idx]);
                  if (n >= round_lower && n < round_lower + WARP_SIZE) {
                    result[t_idx+ni] = adj_buffer[n % WARP_SIZE];
                    // printf("LOW round %d; pref %d; n_idx %d; t_idx %d; old_t %d; t %d\n", r, prefix, n, t_idx+ni, result[old_t_idx], result[t_idx+ni]);
                  }
                }
              }
              __syncwarp();
            }
          } else {
            old_t -= first_med;
            vidType old_t_deg = med_g.get_degree(old_t);
            for (int ni = thread_lane; ni < step_sample_size; ni+=WARP_SIZE) {
              int n = (int)(ceil(warp_idxs[(j * step_sample_size) + ni] * old_t_deg) - 1);
              result[t_idx+ni] = med_g.decode_vbyte_prefix(old_t, n, interval);
              // printf("MED!! n_idx %d; old_deg %d; old_t %d; t %d\n", n, old_t_deg, result[old_t_idx], result[t_idx+ni]);
            }
          }
        }
        __syncwarp();
      }
      old_t_begin = t_begin;
      t_begin += step_count * n_samples;
      __syncthreads();
    }
}

double multilayer_sample(Graph &g, size_t uncomp_mem, size_t total_mem, vector<vidType>& initial, int n_samples, int total_num, int last_step_num, vidType* result, int block_size, int use_subgraphs, int l_deg, int h_deg, vidType prefix_interval, bool *use_uvas) {
    bool uncomp_uva = use_uvas[0];
    bool high_uva = use_uvas[1];
    bool med_uva = use_uvas[2];
    bool low_uva = use_uvas[3];
    int cur_num = initial.size();
    int n_steps = steps();
    vidType *d_result;
    int *step_counts = new int[n_steps + 1];
    int *d_step_counts;
    curandState *d_states;
    double alloc_t, rand_t, sample_t, dealloc_t;
    int v_size = sizeof(vidType);
    int e_size = sizeof(eidType);
    initial[0] = 626;
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }

    int n_block_warps = block_size / WARP_SIZE;
    int total_threads = n_samples * WARP_SIZE;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    int buffer_bytes = block_size * v_size;
    int idxs_bytes = last_step_num * n_block_warps * sizeof(int);
    int smem_bytes = buffer_bytes + idxs_bytes;
    std::cout << "buff bytes " << buffer_bytes << " " << idxs_bytes << std::endl;
    std::cout << "\nNeed " << num_blocks << " blocks for one sample per warp" << std::endl;
    std::cout << "Shared memory per block: " << smem_bytes << " bytes\n\n";

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

    if (use_subgraphs == 2) {
      // get high degree subgraph
      vidType last_high = 0;
      vidType curr_deg = g.get_degree_vbyte(last_high);
      eidType total_degree = curr_deg;
      while (curr_deg > h_deg) {
        last_high++;
        curr_deg = g.get_degree_vbyte(last_high);
        total_degree += curr_deg;
      }
      last_high--;
      total_degree -= curr_deg;
      GraphGPU high_subg(high_uva, g, last_high + 1, total_degree);

      size_t mem_vert = size_t(last_high + 2)*sizeof(eidType);
      size_t mem_edge = size_t(total_degree)*sizeof(vidType);
      size_t mem_graph = mem_vert + mem_edge;
      std::cout << "High deg subgraph: " << (float)mem_graph / (float)1000000000 << "GB; |V| " << last_high + 1 << "\n";

      // get medium degree subgraph
      vidType first_med = last_high + 1;
      vidType last_med = first_med;
      while (g.get_degree_vbyte(last_med) > l_deg) {
        last_med++;
      }
      last_med--;
      GraphGPUCompressed med_subg(first_med, last_med + 1, g.get_degree_vbyte(first_med), g, prefix_interval, med_uva);

      // get low degree subgraph
      vidType first_low = last_med + 1;
      auto g_rptr = g.rowptr_compressed();
      total_degree = g_rptr[g.V()] - g_rptr[first_low];
      GraphGPUCompressed low_subg(low_uva, g, first_low, g.V() - first_low, total_degree);

      // mem_vert = size_t(first_low - last_high)*sizeof(eidType);
      // mem_edge = size_t(g_rptr[first_low] - g_rptr[first_med])*sizeof(vidType);
      // mem_graph = mem_vert + mem_edge;
      // std::cout << "Med deg subgraph: " << (float)mem_graph / (float)1000000000 << "GB; |V| " << first_low - first_med << "\n";

      mem_vert = size_t(g.V() - first_low + 1)*sizeof(eidType);
      mem_edge = size_t(total_degree)*sizeof(vidType);
      mem_graph = mem_vert + mem_edge;
      std::cout << "Low deg subgraph: " << (float)mem_graph / (float)1000000000 << "GB; |V| " << g.V() - first_low << "\n";
      std::cout << "\nLow degree subgraph has max_deg " << prefix_interval - 1 << "; medium degree subgraph has max_deg " << h_deg << std::endl;
      std::cout << "Starting sampling on subgraphs version with " << total_threads << " threads...\n";
      idxs_bytes = last_step_num * n_block_warps * sizeof(float);
      smem_bytes = buffer_bytes + idxs_bytes;
      sample_t = seconds();
      khop_next_3subs<<<num_blocks,block_size,smem_bytes>>>(low_subg, med_subg, high_subg, block_size, l_deg, first_low, prefix_interval, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      sample_t = seconds() - sample_t;
      std::cout << "Done sampling!" << std::endl;
    }
    else if (use_subgraphs == 3) {
      // get uncompressed top graph
      vidType last_uncomp = 0;
      vidType curr_deg = g.get_degree_vbyte(last_uncomp);
      eidType u_total_deg = curr_deg;
      while (u_total_deg < uncomp_mem) {
        last_uncomp++;
        curr_deg = g.get_degree_vbyte(last_uncomp);
        u_total_deg += curr_deg;
      }
      last_uncomp--;
      u_total_deg -= curr_deg;
      // u_total_deg = 0;
      GraphGPU uncomp_subg(uncomp_uva, g, last_uncomp + 1, u_total_deg);
    
      size_t mem_vert = size_t(last_uncomp + 2)*sizeof(eidType);
      size_t mem_edge = size_t(u_total_deg)*sizeof(vidType);
      size_t mem_graph = mem_vert + mem_edge;
      std::cout << "Uncompressed top deg subgraph: " << (float)mem_graph / (float)1000000000 << "GB; |V| " << last_uncomp + 1 << "\n";

      // get high degree subgraph
      auto g_rptr = g._rowptr_compressed();
      size_t mem_left = (total_mem - uncomp_mem) * 3 / 4;
      vidType first_high = last_uncomp + 1;
      // first_high = 0;
      vidType last_high = first_high;
      curr_deg = g.get_degree_vbyte(last_high);
      size_t curr_mem = g_rptr[first_high+1] - g_rptr[first_high];
      while (curr_deg > h_deg && curr_mem < mem_left) {
        last_high++;
        curr_deg = g.get_degree_vbyte(last_high);
        curr_mem += g_rptr[last_high+1] - g_rptr[last_high];
      }
      last_high--;
      std::cout << "high deg cutoff " << g.get_degree_vbyte(first_high) << " " << g.get_degree_vbyte(last_high) << " " << g.get_degree_vbyte(last_high+1) << std::endl;
      std::cout << "mem left " << mem_left << std::endl;
      GraphGPUCompressed high_subg(first_high, last_high + 1, g.get_degree_vbyte(first_high), g, prefix_interval, high_uva);

      // get medium degree subgraph
      vidType first_med = last_high + 1;
      vidType last_med = first_med;
      while (g.get_degree_vbyte(last_med) > l_deg) {
        last_med++;
      }
      last_med--;
      GraphGPUCompressed med_subg(first_med, last_med + 1, g.get_degree_vbyte(first_med), g, prefix_interval, med_uva);

      // get low degree subgraph
      vidType first_low = last_med + 1;
      eidType l_total_deg = g_rptr[g.V()] - g_rptr[first_low];
      mem_vert = size_t(g.V() - first_low + 1)*sizeof(eidType);
      mem_edge = size_t(l_total_deg)*sizeof(vidType);
      mem_graph = mem_vert + mem_edge;
      std::cout << "Low deg subgraph: " << (float)mem_graph / (float)1000000000 << "GB; |V| " << g.V() - first_low << "\n";

      GraphGPUCompressed low_subg(low_uva, g, first_low, g.V() - first_low, l_total_deg);

      // warm_up_gpu<<<num_blocks,block_size>>>(med_subg, total_threads);
      // CUDA_SAFE_CALL(cudaDeviceSynchronize());
      
      std::cout << "\nLow degree subgraph has max_deg " << l_deg << "; medium degree subgraph has max_deg " << h_deg << std::endl;
      std::cout << "Starting sampling on subgraphs version with " << total_threads << " threads...\n";
      idxs_bytes = last_step_num * n_block_warps * sizeof(float);
      smem_bytes = buffer_bytes + idxs_bytes;
      sample_t = seconds();
      std::cout << "initial " << initial[0] << std::endl;
      khop_next_4subs<<<num_blocks,block_size,smem_bytes>>>(low_subg, med_subg, high_subg, uncomp_subg, block_size, l_deg, first_low, first_med, first_high, prefix_interval, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      sample_t = seconds() - sample_t;
      std::cout << "Done sampling!" << std::endl;
    }
    else {
      bool use_uva = false;
      if (use_subgraphs == 1) use_uva = true;
      GraphGPUCompressed gg(g, "streamvbyte", g.get_degree_threshold(), 0, 1, use_uva);
      
      std::cout << "Starting sampling with " << total_threads << " threads...\n";
      sample_t = seconds();
      khop_next<<<num_blocks,block_size,smem_bytes>>>(gg, block_size, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
      // khop_next0<<<num_blocks,block_size>>>(gg, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      sample_t = seconds() - sample_t;
      std::cout << "Done sampling!" << std::endl;
    }
    // if (use_subgraphs)  khop_next3<<<num_blocks,block_size>>>(low_subg, med_subg, high_subg, first_low, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
    // khop_next0<<<num_blocks,block_size>>>(gg, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
    // cudaLaunchCooperativeKernel((void*)(khop_next0), grid, block, kernel_args);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
  

    dealloc_t = seconds();
    CUDA_SAFE_CALL(cudaMemcpy(result, d_result, total_num * v_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_result));
    CUDA_SAFE_CALL(cudaFree(d_states));
    CUDA_SAFE_CALL(cudaFree(d_step_counts));
    dealloc_t = seconds() - dealloc_t;

    std::cout << "Time elapsed for allocating and copying " << alloc_t + dealloc_t << " sec\n\n";

    return sample_t;
}

double multilayer_sample_loaded(Graph &cpu_low, Graph &cpu_med, Graph &cpu_high, Graph &cpu_uncomp, vector<vidType>& initial, int n_samples, int total_num, int last_step_num, vidType* result, int block_size, int use_subgraphs, int l_deg, int h_deg, vidType prefix_interval, bool *use_uvas) {
    bool uncomp_uva = use_uvas[0];
    bool high_uva = use_uvas[1];
    bool med_uva = use_uvas[2];
    bool low_uva = use_uvas[4];
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

    int n_block_warps = block_size / WARP_SIZE;
    int total_threads = n_samples * WARP_SIZE;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    int buffer_bytes = block_size * v_size;
    int idxs_bytes = last_step_num * n_block_warps * sizeof(int);
    int smem_bytes = buffer_bytes + idxs_bytes;
    std::cout << "buff bytes " << buffer_bytes << " " << idxs_bytes << std::endl;
    std::cout << "\nNeed " << num_blocks << " blocks for one sample per warp" << std::endl;
    std::cout << "Shared memory per block: " << smem_bytes << " bytes\n\n";

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

    if (use_subgraphs == 2) {
      vidType first_med = cpu_high.V();
      vidType first_low = first_med + cpu_med.V();
      GraphGPU high_subg(cpu_high, high_uva);
      cpu_high.deallocate();
      GraphGPUCompressed med_subg(cpu_med, "streamvbyte", 0, 0, 1, med_uva);
      cpu_med.deallocate();
      GraphGPUCompressed low_subg(cpu_low, "streamvbyte", 0, 0, 1, low_uva);
      cpu_low.deallocate();

      std::cout << "\nLow degree subgraph has max_deg " << prefix_interval - 1 << "; medium degree subgraph has max_deg " << h_deg << std::endl;
      std::cout << "Starting sampling on subgraphs version with " << total_threads << " threads...\n";
      idxs_bytes = last_step_num * n_block_warps * sizeof(float);
      smem_bytes = buffer_bytes + idxs_bytes;
      sample_t = seconds();
      khop_next_3subs<<<num_blocks,block_size,smem_bytes>>>(low_subg, med_subg, high_subg, block_size, l_deg, first_low, prefix_interval, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      sample_t = seconds() - sample_t;
      std::cout << "Done sampling!" << std::endl;
    }
    else if (use_subgraphs == 3) {
      vidType first_high = cpu_uncomp.V();
      vidType first_med = first_high + cpu_high.V();
      vidType first_low = first_med + cpu_med.V();

      GraphGPU uncomp_subg(cpu_uncomp, uncomp_uva);
      cpu_uncomp.deallocate();
      GraphGPUCompressed high_subg(cpu_high, "streamvbyte", 0, 0, 1, high_uva);
      cpu_high.deallocate();
      GraphGPUCompressed med_subg(cpu_med, "streamvbyte", 0, 0, 1, med_uva);
      cpu_med.deallocate();
      GraphGPUCompressed low_subg(cpu_low, "streamvbyte", 0, 0, 1, low_uva);
      cpu_low.deallocate();

      std::cout << "\nLow degree subgraph has max_deg " << prefix_interval - 1 << "; medium degree subgraph has max_deg " << h_deg - 1 << std::endl;
      std::cout << "First low is " << first_low << " first med is " << first_med << std::endl;
      std::cout << "Starting sampling on subgraphs version with " << total_threads << " threads...\n";
      idxs_bytes = last_step_num * n_block_warps * sizeof(float);
      smem_bytes = buffer_bytes + idxs_bytes;
      warm_up_gpu<<<num_blocks,block_size>>>(med_subg, total_threads);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      sample_t = seconds();
      khop_next_4subs<<<num_blocks,block_size,smem_bytes>>>(low_subg, med_subg, high_subg, uncomp_subg, block_size, l_deg, first_low, first_med, first_high, prefix_interval, d_result, n_steps, n_samples, d_step_counts, total_threads, d_states);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      sample_t = seconds() - sample_t;
      std::cout << "Done sampling!" << std::endl;
    }

    dealloc_t = seconds();
    CUDA_SAFE_CALL(cudaMemcpy(result, d_result, total_num * v_size, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_result));
    CUDA_SAFE_CALL(cudaFree(d_states));
    CUDA_SAFE_CALL(cudaFree(d_step_counts));
    dealloc_t = seconds() - dealloc_t;

    std::cout << "Time elapsed for allocating and copying " << alloc_t + dealloc_t << " sec\n\n";

    return sample_t;
}

void write_subgraphs(Graph &g, size_t uncomp_mem, size_t total_mem, int use_subgraphs, int low_deg, int high_deg, vidType prefix_interval, std::string out_prefix) {
  std::cout << "file name! " << out_prefix << std::endl;
  std::string file_specs = std::to_string(low_deg) + "_" + std::to_string(high_deg);
  // get high degree subgraph

  vidType last_uncomp = 0;
  vidType curr_deg = g.get_degree_vbyte(last_uncomp);
  eidType total_deg = curr_deg;
  while (total_deg < uncomp_mem) {
    last_uncomp++;
    curr_deg = g.get_degree_vbyte(last_uncomp);
    total_deg += curr_deg;
  }
  last_uncomp--;
  total_deg -= curr_deg;
  GraphGPU uncomp_subg(false, g, last_uncomp + 1, total_deg, out_prefix + "u" + std::to_string(total_mem / uncomp_mem));
  std::cout << "Allocated uncompressed subgraph\n";

  // get high degree subgraph
  auto g_rptr = g.rowptr_compressed();
  size_t mem_left = (total_mem - uncomp_mem) * 3 / 4;
  vidType first_high = last_uncomp + 1;
  vidType last_high = first_high;
  curr_deg = g.get_degree_vbyte(last_high);
  size_t curr_mem = g_rptr[first_high+1] - g_rptr[first_high];
  while (curr_mem < mem_left && curr_deg > high_deg) {
    last_high++;
    curr_deg = g.get_degree_vbyte(last_high);
    curr_mem += g_rptr[last_high+1] - g_rptr[last_high];
  }
  last_high--;
  std::cout << "high deg cutoff " << g.get_degree_vbyte(first_high) << " " << g.get_degree_vbyte(last_high) << " " << g.get_degree_vbyte(last_high+1) << std::endl;
  std::cout << "mem left " << mem_left << std::endl;
  GraphGPUCompressed high_subg(first_high, last_high + 1, g.get_degree_vbyte(first_high), g, prefix_interval, false, out_prefix + "h" + std::to_string(high_deg));
  std::cout << "Allocated high subgraph\n";

  // get medium degree subgraph
  vidType first_med = last_high + 1;
  vidType last_med = first_med;
  while (g.get_degree_vbyte(last_med) > low_deg) {
    last_med++;
  }
  last_med--;
  GraphGPUCompressed med_subg(first_med, last_med + 1, g.get_degree_vbyte(first_med), g, prefix_interval, true, out_prefix + "m" + file_specs);
  std::cout << "Allocated medium subgraph\n";

  // get low degree subgraph
  vidType first_low = last_med + 1;
  // auto g_rptr = g.rowptr_compressed();
  total_deg = g_rptr[g.V()] - g_rptr[first_low];
  size_t mem_vert = size_t(g.V() - first_low + 1)*sizeof(eidType);
  size_t mem_edge = size_t(total_deg)*sizeof(vidType);
  size_t mem_graph = mem_vert + mem_edge;
  std::cout << "Low deg subgraph: " << (float)mem_graph / (float)1000000000 << "GB; |V| " << g.V() - first_low << "\n";

  GraphGPUCompressed low_subg(true, g, first_low, g.V() - first_low, total_deg, out_prefix + "l" + std::to_string(low_deg));
  std::cout << "Allocated low subgraph\n";
  std::cout << "First low is " << first_low << " first med is " << first_med << std::endl;
}

int main(int argc, char* argv[]) {
  // size_t memsize = print_device_info(0);
  std::string in_prefix = argv[1];
  std::string out_prefix = argv[2];
  // std::string out_prefix = argv[2];
  std::string scheme = "streamvbyte";
  bool permutated = false;
  bool compress_graph = false;
  int c;
  bool print = false;
  int n_samples = num_samples();
  int pdeg = BLOCK_SIZE;
  int low_deg = 32;
  int high_deg = 256;
  int use_subgraphs = 0; // 0 = in mem no subgraphs, 1 = uva no subgraphs, 2 = use subgraphs, 3 = high subgraph uses prefix
  bool write_subs = false;
  bool read_subs = false;
  vidType prefix_interval = WARP_SIZE;
  size_t gpu_mem = 10000000000;
  int uncomp_mem_ratio = 4;
  size_t gpu_mem_uncomp = gpu_mem / uncomp_mem_ratio;
  while ((c = getopt(argc, argv, "wrcn:d:l:h:s:v:")) != -1) {
    switch (c) {
      case 'w':
        write_subs = true;
        break;
      case 'r':
        read_subs = true;
        break;
      case 'c':
        compress_graph = true;
        break;
      case 'n':
        n_samples = atoi(optarg);
        break;
      case 'd':
        pdeg = atoi(optarg);
        break;
      case 'l':
        low_deg = atoi(optarg);
        break;
      case 'h':
        high_deg = atoi(optarg);
        break;
      case 's':
        use_subgraphs = atoi(optarg);
        break;
      case 'v':
        prefix_interval = (vidType)atoi(optarg);
        break;
      default:
        abort();
    }
  }
  bool uva_flags[4];
  std::ifstream file("/home/mcai1/GraphAIBench/src/compressing/config.txt");
  std::string str; 
  for (int i = 0; i < 4; i++) {
    std::getline(file, str);
    if (str == "true") uva_flags[i] = true;
    else uva_flags[i] = false;
    std::cout << uva_flags[i] << " " << str << std::endl;
  }
  if (compress_graph) { 
    save_compressed_graph_vbyte(in_prefix, out_prefix); 
    return 0;
  }
  if (prefix_interval % 16 != 0) {
    std::cout << "QUITTING; PREFIX INTERVAL MUST BE MULTIPLE OF 16\n";
    return 0;
  }
  if (prefix_interval > high_deg) {
    std::cout << "QUITTING; LOW DEGREE MUST BE LEQ HIGH DEGREE\n";
    return 0;
  }
  // g.print_meta_data();
  std::cout << "LOADED COMPRESSED GRAPH\n" << std::endl;
  if (write_subs) {
    Graph g;
    g.load_compressed_graph(in_prefix, scheme, permutated);
    write_subgraphs(g, gpu_mem_uncomp, gpu_mem, use_subgraphs, low_deg, high_deg, prefix_interval, out_prefix);
    return 0;
  }

  // int n_samples = argc >= 4 ? atoi(argv[3]) : num_samples();
  // int pdeg = argc >= 5 ? atoi(argv[4]) : BLOCK_SIZE;
  std::cout << "block size: " << pdeg << " high deg: " << high_deg << " low deg: " << low_deg << "\n";

  double iElaps;
  int total_count;
  vidType* result;
  if (read_subs) {
    std::string end_prefix = std::to_string(low_deg) + "_" + std::to_string(high_deg);
    Graph cpu_uncomp;
    Graph cpu_high;
    if (use_subgraphs == 2) cpu_high.load_graph(in_prefix + "h" + std::to_string(high_deg));
    else if (use_subgraphs == 3) {
      cpu_uncomp.load_graph(in_prefix + "u" + std::to_string(uncomp_mem_ratio));
      cpu_high.load_compressed_graph(in_prefix + "h" + std::to_string(high_deg), "streamvbyte", false);
    }
    Graph cpu_med;
    cpu_med.load_compressed_graph(in_prefix + "m" + end_prefix, "streamvbyte", false);
    Graph cpu_low;
    cpu_low.load_compressed_graph(in_prefix + "l" + std::to_string(low_deg), "streamvbyte", false);
    vector<vidType> initial = get_initial_transits(sample_size(-1) * n_samples, cpu_high.V() + cpu_med.V() + cpu_low.V());
    // initial[0] = 32317;
    std::cout << "total v " << cpu_high.V() + cpu_med.V() + cpu_low.V() << "\n";
    int step_count = sample_size(-1) * n_samples;
    total_count = step_count;
    for (int step = 0; step < steps(); step++) {
      step_count *= sample_size(step);
      total_count += step_count;
    }
    result = new vidType[total_count];
    std::fill_n(result, total_count, MAX_VIDTYPE);
    iElaps = multilayer_sample_loaded(cpu_low, cpu_med, cpu_high, cpu_uncomp, initial, n_samples, total_count, step_count / n_samples, result, pdeg, use_subgraphs, low_deg, high_deg, prefix_interval, uva_flags);
  }
  else {
    Graph g;
    g.load_compressed_graph(in_prefix, scheme, permutated);
    vector<vidType> initial = get_initial_transits(sample_size(-1) * n_samples, g.V());
    // initial[0] = 32317;
    int step_count = sample_size(-1) * n_samples;
    total_count = step_count;
    for (int step = 0; step < steps(); step++) {
      step_count *= sample_size(step);
      total_count += step_count;
    }
    result = new vidType[total_count];
    std::fill_n(result, total_count, MAX_VIDTYPE);
    iElaps = multilayer_sample(g, gpu_mem_uncomp, gpu_mem, initial, n_samples, total_count, step_count / n_samples, result, pdeg, use_subgraphs, low_deg, high_deg, prefix_interval, uva_flags);
  }

  std::cout << "Sampled total of " << total_count << " transits in " << steps() << " steps\n";
  std::cout << "Time elapsed for sampling " << iElaps << " sec\n\n";
  std::cout << "\n";
  delete[] result;

  return 0;
}
