#include "graph.h"

void triangle_count_compressed(Graph &g, uint64_t &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting on Compressed Graph (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  //auto md = g.get_max_degree();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < g.V(); u ++) {
    VertexSet adj_u(u); // u's adjacency list
    vidType deg_u = g.decode_vertex(u, adj_u.data());
    adj_u.adjust_size(deg_u);
    adj_u.sort();
    for (auto v : adj_u) {
      if (v > u) continue;
      VertexSet adj_v(v); // v's adjacency list
      vidType deg_v = g.decode_vertex(v, adj_v.data());
      adj_v.adjust_size(deg_v);
      adj_v.sort();
      auto num = (uint64_t)intersection_num(adj_u, adj_v, v);
      counter += num;
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [tc_omp_compressed] = " << t.Seconds() << " sec\n";
  return;
}

void triangle_count(Graph &g, uint64_t &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting (" << num_threads << " threads)\n";
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < g.V(); u ++) {
    auto adj_u = g.N(u);
    for (auto v : adj_u) {
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [tc_omp_base] = " << t.Seconds() << " sec\n";
  return;
}

