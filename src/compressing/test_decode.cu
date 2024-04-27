#include "graph.h"
#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cmath>
#include "graph_gpu_compressed.h"
#include "compressor.hh"
using namespace std;
using namespace cooperative_groups;

__global__ void test_warp_decompress(GraphGPUCompressed g, int total_threads, vidType *buffer, int n_idx) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = thread_id / WARP_SIZE;
  if (thread_id >= total_threads) {
    return;
  }
  // vidType v_id = (vidType) warp_id;
  vidType v_id = 1;
  vidType deg = g.get_degree(v_id);
  vidType *adj = buffer + (g.get_max_degree() * v_id);
  if (threadIdx.x % WARP_SIZE < 1) {
    g.decode_vbyte_sums(v_id, adj, n_idx);
  }
  // if (threadIdx.x % WARP_SIZE < deg) {
  //   g.decode_vbyte_warp<scheme,delta,pack_size>(v_id, adj);
  // }
}

__global__ void print_buffer(GraphGPUCompressed g, vidType *buffer, int num, int n_idx) {
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id >= 1) {
    return;
  }
  for (int i = 0; i < num; i++) {
    i = 1;
    printf("v_id %d; deg %d\n", i, g.get_degree(i));
    vidType *adj = buffer + (g.get_max_degree() * i);
    // printf("v_id %d; adj %d %d %d %d\n", i, adj[0], adj[1], adj[2], adj[3]);
    // printf("v_id %d; deg %d, adj[0] %d\n", i, g.get_degree(i), adj[0]);

    // for (int j = 0; j < g.get_degree(i); j++) {
    //   printf("n_id%d: %d ", j, adj[j]);
    // }

    printf("n_id%d: %d\n", n_idx, adj[0]);
  }
}

void move_onto_gpu(Graph &g, int n_idx) {
  GraphGPUCompressed gg(g, "streamvbyte", g.get_degree_threshold(), 0, 1, true);
  vidType *buffer;
  vidType max_degree = g.get_max_degree();
  // int num = g.V();
  int num = 1;
  int block_size = 256;
  size_t warps_per_block = block_size / WARP_SIZE;
  size_t nblocks = (num-1)/warps_per_block+1;
  allocate_gpu_buffer(size_t(max_degree) * warps_per_block * nblocks, buffer);
  int threads = num * WARP_SIZE;
  int num_blocks = (threads + block_size - 1) / block_size;
  test_warp_decompress<<<num_blocks,block_size>>>(gg, threads, buffer, n_idx);
  cudaDeviceSynchronize();
  print_buffer<<<1,1>>>(gg, buffer, num, n_idx);
  cudaDeviceSynchronize();
  cudaFree(buffer);
}

int main(int argc, char* argv[]) {
  // Graph g;
  std::string in_prefix = argv[1];
  std::string out_prefix = argv[2];
  std::string scheme = "streamvbyte";
  bool permutated = false;
  bool compress_graph = false;
  int n_idx = 0;
  int c;
  int idx = 0;
  vidType transit = 0;
  while ((c = getopt(argc, argv, "cs:t:i:")) != -1) {
    switch (c) {
      case 'c':
        compress_graph = true;
        break;
      case 's':
        n_idx = atoi(optarg);
        break;
      case 't':
        transit = (vidType)atoi(optarg);
        break;
      case 'i':
        idx = atoi(optarg);
        break;
      default:
        abort();
    }
  }
  // if (compress_graph) { save_compressed_graph(in_prefix, out_prefix); }
  // g.load_compressed_graph(out_prefix, scheme, permutated);
  Graph g(in_prefix, 0, 0, 0, 0, 0);
  // std::cout << "deg " << g.get_degree(transit) << "\nneighbor " << g.N(transit, idx) << std::endl;
  // move_onto_gpu(g, n_idx);

  // count number of low, med, and high degree nodes
  // int low, med, high = 0;
  // int low_threshold = 32;
  // int high_threshold = 64;
  // for (int v = 0; v < g.V(); v++) {
  //   int deg = g.get_degree(v);
  //   if (deg <= low_threshold) {
  //     low++;
  //   }
  //   else
  //   else if (deg > high_threshold) {
  //     high++;
  //   }
  //   else {
  //     med++;
  //   }
  // }
  // std::cout << "num of low(<=" << low_threshold << "): " << low << "\n";
  // std::cout << "num of high(>" << high_threshold << "): " << high << "\n";
  // std::cout << "num of med: " << med << "\n";

  int num_blocks = 33;
  vidType blocks[num_blocks] = {0};
  eidType num_neighbors_blocks[num_blocks] = {0};
  int block_size = 32;
  for (int v = 0; v < g.V(); v++) {
    int deg = g.get_degree(v);
    int idx;
    if (deg % block_size == 0 && deg != 0) {idx = min(deg / block_size - 1, num_blocks - 1);}
    else {
      idx = min(deg / block_size, num_blocks - 1);
    }
    blocks[idx]++;
    num_neighbors_blocks[idx] += deg;
  }
  //for reading here
  // for (int i = 0; i < num_blocks; i++) {
  //   if (i == 0) {
  //     std::cout << i * 32 << "-" << (i+1) * 32 << ": " << blocks[i] << " vertices; " << num_neighbors_blocks[i] << " neighbors\n";
  //   } else if (i == num_blocks - 1) {
  //     std::cout << i * 32 + 1 << "+: " << blocks[i] << " vertices; " << num_neighbors_blocks[i] << " neighbors\n";
  //   } else {
  //     std::cout << i * 32 + 1 << "-" << (i+1) * 32 << ": " << blocks[i] << " vertices; " << num_neighbors_blocks[i] << " neighbors\n";
  //   }
  // }
  //for copying into sheets
  for (int i = 0; i < num_blocks; i++) {
    std::cout << blocks[i] << "\n";
  }
  std::cout << "\n\n";
  for (int i = 0; i < num_blocks; i++) {
    std::cout << num_neighbors_blocks[i] << "\n";
  }
}