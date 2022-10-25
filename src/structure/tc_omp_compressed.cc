#include "graph.h"

void triangle_count_compressed(Graph &g, uint64_t &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting on Compressed Graph (" << num_threads << " threads)\n";
  timers[SETOPS] = 0.0;
  timers[DECOMPRESS] = 0.0;
  bool need_order = true;
  Timer t;
  t.Start();
  uint64_t counter = 0;
  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
  for (vidType u = 0; u < g.V(); u ++) {
    //VertexSet adj_u(u); // u's adjacency list
    //g.decode_vertex(u, adj_u, USE_INTERVAL);
    auto adj_u = g.N_compressed(u, need_order);
    for (auto v : adj_u) {
      //if (v > u) break;
      //auto num = (uint64_t)g.intersect_num_compressed(adj_u, v, v);
      auto num = (uint64_t)g.intersect_num_compressed(adj_u, v);
      counter += num;
    }
  }
  t.Stop();
  std::cout << "runtime [tc_omp_compressed_opt] = " << t.Seconds() << " sec\n";
  std::cout << "total_num_triangles = " << counter << "\n";
  std::cout << "Set operations time: "   << timers[SETOPS] << "\n";
  std::cout << "Decompress time: "   << timers[DECOMPRESS] << "\n";
  total = counter;
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
      //if (v > u) break;
      //counter += (uint64_t)intersection_num(adj_u, g.N(v), v);
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
    }
  }
  total = counter;
  t.Stop();
  std::cout << "runtime [tc_omp_base] = " << t.Seconds() << " sec\n";
  return;
}

