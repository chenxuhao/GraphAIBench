// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "ann.h"

void ANN(Graph &g, std::vector<Embedding> points, Embedding query) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP ANN (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  #pragma omp parallel for
  for (vidType u = 0; u < g.V(); u ++) {
    auto adj_u = g.N(u);
    // add code here
  }
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
  return;
}

