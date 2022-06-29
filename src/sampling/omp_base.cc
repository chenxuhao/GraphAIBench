// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void sample(Graph &g, Graph &subg) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Graph Sampling (" << num_threads << " threads)\n";
  //auto nv = g.V();
  Timer t;
  t.Start();
  VertexList frontier_in, frontier_out;
  int num_hops = 3;
  int iter = 0;
  for (; iter < num_hops; iter ++) {
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < frontier_in.size(); i++) {
      auto v = frontier_in[i];
      for (auto u : g.N(v)) {
        frontier_out.push_back(u);
      }
    }
    std::cout << "Iterations " << iter+1 << ": output frontier size = " << frontier_out.size() << "\n";
    std::swap(frontier_in, frontier_out);
  }
  t.Stop();
  std::cout << "iterations = " << iter+1 << ".\n";
  std::cout << "runtime [sample_omp_base] = " << t.Seconds() << " sec\n";
  return;
}

