// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void KCoreSolver(Graph &g, std::vector<int> &coreness, vidType &largest_core, int n_gpu, int chunk_size);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> \n";
    std::cout << "Example: " << argv[0] << " ../inputs/citeseer/graph\n";
    exit(1);
  }
  std::cout << "K-Core decomposition: assumes symmetric graph\n";
  Graph g(argv[1]); // use DAG
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 2) n_devices = atoi(argv[2]);
  if (argc > 3) chunk_size = atoi(argv[3]);
  g.print_meta_data();
  std::vector<int> coreness(g.V(), 0);
  vidType largest_core = 0;
  KCoreSolver(g, coreness, largest_core, n_devices, chunk_size);
  std::cout << "largestCore = " << largest_core<< "\n";
  return 0;
}

