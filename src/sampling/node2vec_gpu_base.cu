#include "graph_gpu.h"
#include "cuda_launch_config.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <random>

#define max(a, b) (((a) > (b)) ? (a) : (b))

__global__ void node2vec_walk(GraphGPU g, curandState* states, int *sources, int *paths, int numSources, int hops, float p, float q) {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  if(id >= numSources) {
    return ;
  }

  // faster access in the local memory
  curandState localState = states[id];

  int node = sources[id];
  int deg = g.getOutDegree(node);

  int prev = -1;
  float max_prob = max(1/q, max(1, 1/p));
  int k = 0;

  while(deg && k < hops) {
    int i = ((int) floor(curand_uniform(&localState) * deg * 2)) % deg;
    int child = g.N(node, i);
    float prob = 1 / q;
    if(prev == child) {
      prob = 1 / p;
    } else if(prev != -1) {
      for(int j = 0; j < g.getOutDegree(prev); j++) {
        if(g.N(prev, j) == child) {
          prob = 1.0; 
        }
      }
    }
    if(max_prob * curand_uniform(&localState) <= prob) {
      paths[hops * id + k] = child;    
      node = child;
      k += 1;
    }
  }

  // update the state in the global memory
  states[id] = localState;
}

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

 
int* node2vec(Graph &hostG, std::vector <int> sources, int hops) {
  GraphGPU g (hostG);

  std::cout << "Number of sources: " << sources.size() << std::endl;
  
  const int numSources = sources.size();
  const int threadsPerBlock = 16;
  const int numBlocks = (numSources + threadsPerBlock - 1) / threadsPerBlock;
  const int numThreads = numBlocks * threadsPerBlock;

  const int size = sizeof(int) * hops * numSources;
  int *hostPaths = (int*) malloc(size);
  int *hostSources = (int*) malloc(sizeof(int) * numSources);
  int *devSources;
  int *devPaths;
  curandState *devStates;

  for(int i = 0; i < numSources; i++) {
    hostSources[i] = sources[i];
  }

  CUDA_SAFE_CALL(cudaMalloc((void **)&devSources, sizeof(int) * numSources));  

  CUDA_SAFE_CALL(cudaMalloc((void **)&devStates, numThreads *
                  sizeof(curandState)));
  
  CUDA_SAFE_CALL(cudaMalloc((void**) &devPaths, size));
  
  // copy over the sources to the device
  CUDA_SAFE_CALL(cudaMemcpy(devSources, hostSources, sizeof(int) * numSources, cudaMemcpyHostToDevice));
    
  setup_kernel<<<numBlocks, threadsPerBlock>>>(devStates);

  node2vec_walk<<<numBlocks, threadsPerBlock>>>(g, devStates, devSources, devPaths, numSources, hops, 0.3, 0.2);
  
  CUDA_SAFE_CALL(cudaMemcpy(hostPaths, devPaths, size, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(devPaths));
  CUDA_SAFE_CALL(cudaFree(devSources));
  CUDA_SAFE_CALL(cudaFree(devStates));
  free(hostSources);

  return hostPaths;
}

