// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph_partition.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "Test graph partitioning.\n";
  int n_devices = 2;
  if (argc > 2) n_devices = atoi(argv[2]);
  Graph g(argv[1], 1);
  g.print_meta_data();
  PartitionedGraph pg(&g, n_devices);
  pg.edgecut_induced_partition1D();
  //pg.print_subgraphs();

  Timer t;
  t.Start();
  uint64_t counter = 0;
  for (int i = 0; i < pg.get_num_subgraphs(); i++) {
    auto sg = pg.get_subgraph(i);
    auto begin = pg.get_local_begin(i);
    auto end = pg.get_local_end(i);
    std::cout << "subgraph[" << i << "]: from vertex " << begin << " to " << end << "\n";
    #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
    for (vidType u = begin; u < end; u ++) {
      auto yu = sg->N(u);
      for (auto v : yu) {
        counter += (uint64_t)intersection_num(yu, sg->N(v));
      }
    }
  }
  t.Stop();
  std::cout << "total_num_triangles = " << counter << "\n";
  std::cout << "runtime [tc_omp_base] = " << t.Seconds() << " sec\n";
  return 0;
}

