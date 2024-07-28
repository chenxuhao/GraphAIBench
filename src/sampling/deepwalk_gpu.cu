#include <iostream>
#include <fstream>
#include "graph.h"
#include "graph_gpu.h"
#include "deepwalk.h"
using namespace std;

#define SAMPLE_SIZE 1 // fanout of the random walk

// The DeepWalk algorithm runs RandomWalk (serial) \gamma times from each vertex, generating a total of \gamma * |V| walks.
// This parallelizes over different samples, not over different steps of the same random walk.
__global__ RandomWalkKernel(GraphGPU &g, vector<vector<vidType>> &device_all_transits, vector<vidType> &all_vertices, int n_samples=1, int steps, int gamma) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= g.V() * gamma) return 0;

    // all_transits keeps track of the sampled nodes at each step of the algorithm
    vidType initial_vertex = all_vertices[blockIdx.x];
    vector<vidType> all_transits = {initial_vertex};

    int step_count = SAMPLE_SIZE * n_samples; // step_count = prev_step_count = 1
    int prev_step_count = n_samples;
    int t_begin = 0;
    int old_t_begin = 0;
    for (int step = 0; step < steps; step++) {
      t_begin += step_count; // t_begin++
      step_count *= SAMPLE_SIZE;
      prev_step_count *= SAMPLE_SIZE; // step_count and prev_step_count do not change; i.e. they are always one

      // assume individual transit sampling - otherwise, we would consider whether samplingType() is Individual or Collective
      for (int idx = 0; idx < step_count; idx++) { // idx = 0 is the only option if step_count = 1
        int t_idx = t_begin + idx; // t_idx = t_begin
        int old_t_idx = old_t_begin + idx / SAMPLE_SIZE; // old_t_idx = old_t_begin
        vidType old_t = all_transits[old_t_idx]; // old_t = previous transit vertex
        if (old_t == (numeric_limits<uint32_t>::max)()) { 
          all_transits[t_idx] = (numeric_limits<uint32_t>::max)();
          continue;
        }
        vidType old_t_degree = g.out_degree(old_t);
        vidType new_t = (numeric_limits<uint32_t>::max)();
        if (old_t_degree != 0) { 
          new_t = sample_next(g, old_t, old_t_degree, step);
        }
        all_transits[t_idx] = new_t; // the new node (new_t) is stored in all_transits at index t_idx
      }

    old_t_begin += prev_step_count;
  }
  __syncthreads();
  device_all_transits.push_back(all_transits);

  return 0;
}

/**
 * @param g is a reference to the graph 
 * @param steps is the length of each random walk
 * @param gamma is the number of random walks taken per node
*/
int RandomWalk(Graph &g, int n_samples=1, int n_threads, int steps, int gamma) {
    cout << "CUDA Graph Sampling" << endl;

    Timer t;
    t.Start();
    vector<vector<vidType>> * host_output;
    vector<vidType> * all_vertices;
    * all_vertices = get_initial_transits(g.V(), g.V());

    // Allocate memory on the GPU
    cout << "Allocating Memory on the GPU" << endl;
    Graph * device_g;
    vector<vector<vidType>> * device_all_transits; // device_all_transits is a vector of all random walks
    cudaMalloc((void**) &device_g, sizeof(Graph)); // ??
    cudaMalloc((void**) &all_vertices, vid_size * g.V());
    cudaMalloc((void**) &device_all_transits, vid_size * g.V() * gamma); // g.V() returns number of vertices

    // Copy data to the GPU
    cout << "Copying Data to the GPU" << endl;
    cudaMemcpy(device_g, g, sizeof(Graph), cudaMemcpyHostToDevice); // ??

    // Launch the kernel
    cout << "Launching the Kernel" << endl;
    dim3 DimGrid(g.V(), 1, 1); // # blocks in grid
    dim3 DimBlock(gamma, 1, 1); // # threads per block
    RandomWalkKernel<<<DimGrid, DimBlock>>>(device_g, device_all_transits, all_vertices, steps, gamma);

    // Copy results back to CPU
    cout << "Copying Results Back to CPU" << endl;
    cudaMemcpy(host_output, device_all_transits, vid_size * g.V() * gamma, cudaMempyDeviceToHost);

    // Free GPU Memory
    cout << "Freeing GPU Memory" << endl;
    cudaFree(device_g);
    cudaFree(device_all_transits);

    t.Stop();
    std::cout << "Finished sampling in " << t.Seconds() << " sec" << endl;

    return 0;
};