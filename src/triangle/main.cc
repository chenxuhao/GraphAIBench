// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void TCSolver(Graph &g, uint64_t &total, int n_gpu, int chunk_size);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [oriented(0)] [partitioned(0)] [num_gpu(1)] [chunk_size(1024)] [adj_sorted(1)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph\n";
    exit(1);
  }

  std::cout << "Triangle Counting: assuming the neighbor lists are sorted.\n";
  int oriented = 0;
  if (argc > 2) oriented = atoi(argv[2]);
  bool need_orientation = true;
  if (oriented) need_orientation = false;
  int is_partitioned = 0;
  if (argc > 3) is_partitioned = atoi(argv[3]);
  Graph g(argv[1], need_orientation, oriented, 0, 0, 0, 0, is_partitioned);

  int n_devices = 1;
  int chunk_size = 1024;
  int adj_sorted = 1;
  if (argc > 4) n_devices = atoi(argv[4]);
  if (argc > 5) chunk_size = atoi(argv[5]);
  g.print_meta_data();
  if (argc > 6) adj_sorted = atoi(argv[6]);
  if (!adj_sorted) g.sort_neighbors();
  uint64_t total = 0;
  TCSolver(g, total, n_devices, chunk_size);
  std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}

