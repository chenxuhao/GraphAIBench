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

template <int scheme = 0, bool delta = true, int pack_size = 4>
__global__ void test_warp_decompress(GraphGPUCompressed g, int total_threads, vidType *buffer) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = thread_id / WARP_SIZE;
    if (thread_id >= total_threads) {
        return;
    }
    vidType v_id = (vidType) warp_id;
    vidType deg = g.get_degree(v_id);
    vidType *adj = buffer + (g.get_max_degree() * v_id);
    for (int i = thread_id % WARP_SIZE; i < deg; i += WARP_SIZE) {
        g.decode_vbyte_warp<scheme,delta,pack_size>(v_id, adj);
    }
}

__global__ void print_buffer(GraphGPUCompressed g, vidType *buffer) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= 1) {
        return;
    }
    for (int i = 0; i < g.V(); i++) {
        vidType *adj = buffer + (g.get_max_degree() * i);
        printf("v_id %d; adj %d %d %d %d\n", i, adj[0], adj[1], adj[2], adj[3]);
    }
}

void move_onto_gpu(Graph &g) {
    GraphGPUCompressed gg(g, "streamvbyte", g.get_degree_threshold(), 0, 1, true);
    vidType *buffer;
    vidType max_degree = g.get_max_degree();
    int block_size = 256;
    size_t warps_per_block = block_size / WARP_SIZE;
    size_t nblocks = (g.V()-1)/warps_per_block+1;
    allocate_gpu_buffer(size_t(max_degree) * warps_per_block * nblocks, buffer);
    int threads = g.V() * WARP_SIZE;
    int num_blocks = (threads + block_size - 1) / block_size;
    test_warp_decompress<<<num_blocks,block_size>>>(gg, threads, buffer);
    cudaDeviceSynchronize();
    print_buffer<<<1,1>>>(gg, buffer);
    cudaDeviceSynchronize();
    cudaFree(buffer);
}

int main(int argc, char* argv[]) {
    Graph g;
    std::string in_prefix = argv[1];
    std::string out_prefix = argv[2];
    std::string scheme = "streamvbyte";
    bool permutated = false;
    // save_compressed_graph(in_prefix, out_prefix);
    g.load_compressed_graph(out_prefix, scheme, permutated);
    move_onto_gpu(g);
}