#include <iostream>
#include <fstream>
#include <omp.h>
#include "graph.h"
#include "deepwalk.h"
using namespace std;

#define SAMPLE_SIZE 1 // fanout of the random walk
#define BLOCK_SIZE 256

// The DeepWalk algorithm runs RandomWalk \gamma times from each vertex, generating a total of \gamma * |V| walks.

__global__ RandomWalkKernel(Graph &g, vector<vidType> &device_all_transits, int steps) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int step_count;
    __shared__ int prev_step_count;
    __shared__ int t_begin;
    __shared__ int old_t_begin;

    if (index == 0) {
        step_count = SAMPLE_SIZE * n_samples; // step_count = prev_step_count = 1
        prev_step_count = n_samples;
        t_begin = 0;
        old_t_begin = 0;
        for (int step = 0; step < steps; step++) {
            t_begin += step_count; // t_begin++
            step_count *= SAMPLE_SIZE;
            prev_step_count *= SAMPLE_SIZE; // step_count and prev_step_count do not change
        }
    }

    // the following is from the parallel implementation on CPU
    if (index < steps) {
        t_begin += step_count; // t_begin++
        step_count *= SAMPLE_SIZE;
        prev_step_count *= SAMPLE_SIZE; // step_count and prev_step_count do not change

        // assume individual transit sampling - otherwise, we would consider whether samplingType() is Individual or Collective
        #pragma omp parallel for
        for (int idx = 0; idx < step_count; idx++) { // idx = 0 is the only option if step_count = 1
            int t_idx = t_begin + idx; // t_idx = t_begin
            int old_t_idx = old_t_begin + idx / SAMPLE_SIZE; // old_t_idx = old_t_begin
            vidType old_t = device_all_transits[old_t_idx]; // old_t = previous transit vertex
            if (old_t == (numeric_limits<uint32_t>::max)()) { 
                device_all_transits[t_idx] = (numeric_limits<uint32_t>::max)();
                continue;
            }
            vidType old_t_degree = g.out_degree(old_t);
            vidType new_t = (numeric_limits<uint32_t>::max)();
            if (old_t_degree != 0) { 
                new_t = sample_next(g, old_t, old_t_degree, step);
            }
            device_all_transits[t_idx] = new_t; // the new node (new_t) is stored in all_transits at index t_idx
        }
        old_t_begin += prev_step_count;
    }
}

/**
 * @param g is a reference to the graph 
 * @param initial_vertex is the first vertex of the walk
 * @param steps is the length of each random walk
*/
int RandomWalk(Graph &g, vidType initial_vertex, int n_samples=1, int n_threads, int steps) {
    int num_threads = 1;
    omp_set_num_threads(n_threads);
  
    cout << "CUDA Graph Sampling" << endl;

    Timer t;
    t.Start();
    // all_transits keeps track of the sampled nodes at each step of the algorithm
    vector<vidType> * host_input = {initial_vertex};
    vector<vidType> * host_output;

    // Allocate memory on the GPU
    cout << "Allocating Memory on the GPU" << endl;
    Graph * device_g;
    vector<vidType> * device_all_transits;
    cudaMalloc((void**) &device_g, sizeof(Graph)); // ??
    cudaMalloc((void**) &device_all_transits, vid_size * g.V()); // g.V() returns number of vertices

    // Copy data to the GPU
    cout << "Copying Data to the GPU" << endl;
    cudaMemcpy(device_g, g, sizeof(Graph), cudaMemcpyHostToDevice); // ??
    cudaMemcpy(device_all_transits, all_transits, vid_size * g.V(), cudaMempyHostToDevice);

    // Launch the kernel
    cout << "Launching the Kernel" << endl;
    dim3 DimGrid((steps - 1) / BLOCK_SIZE + 1, 1, 1); // # blocks in grid
    dim3 DimBlock(BLOCK_SIZE, 1, 1); // # threads per block
    RandomWalkKernel<<<DimGrid, DimBlock>>>(device_g, device_all_transits, steps);

    // Copy results back to CPU
    cout << "Copying Results Back to CPU" << endl;
    cudaMemcpy(host_output, device_all_transits, vid_size * g.V(), cudaMempyDeviceToHost);

    // Free GPU Memory
    cout << "Freeing GPU Memory" << endl;
    cudaFree(device_g);
    cudaFree(device_all_transits);

    t.Stop();

    std::cout << "result size: " << step_count + t_begin << endl;
    std::cout << "Finished sampling in " << t.Seconds() << " sec" << endl;

    return 0;
};