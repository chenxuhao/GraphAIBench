// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <omp.h>

// roots: root vertices, from which the sampling starts
void sample(Graph &g, VertexList roots, Graph &subg) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Graph Sampling (" << num_threads << " threads)\n";
  //auto nv = g.V();
  Timer t;
  t.Start();
  VertexLists frontiers;
  int num_hops = 3;
  int sample_size[3] = {15, 10, 10};
  frontiers.resize(num_hops+1);
  frontiers[0] = roots;
  std::vector<size_t> frontier_size(num_hops);
  frontier_size[0] = roots.size();
  for (int i = 0; i < num_hops; i ++) {
    frontier_size[i] *= sample_size[i];
    frontiers[i+1].resize(frontier_size[i]);
  }
  int iter = 0;
  for (; iter < num_hops; iter ++) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < frontiers[iter].size(); i++) {
      auto v = frontiers[iter][i];
      auto degree = g.get_degree(v);
      for (int j = 0; j < sample_size[iter]; j++) {
        int e = rand() % degree; // randomly select a neighbor
        auto u = g.N(v, e);
        frontiers[iter+1][i*sample_size[iter]+j] = u;
      }
    }
    std::cout << "Iterations " << iter+1 << ": output frontier size = " << frontiers[iter+1].size() << "\n";
  }
  t.Stop();
  std::cout << "iterations = " << iter+1 << ".\n";
  std::cout << "runtime [sample_omp_base] = " << t.Seconds() << " sec\n";
  return;
}

