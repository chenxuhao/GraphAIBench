#include "decoder.h"

void triangle_count(CompressedGraph &g, uint64_t &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP TC (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < g.V(); u ++) {
    auto deg = g.get_degree(u);
    for (vidType i = 0; i < degree; i ++) {
      auto v = g.N(u, i);
      counter += (uint64_t)intersection_num(u, v);
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [tc_omp_base] = " << t.Seconds() << " sec\n";
  return;
}

