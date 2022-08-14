// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void CCSolver(Graph &g, comp_t *comp);
void CCVerifier(Graph &g, comp_t *comp_test);

int main(int argc, char *argv[]) {
  std::cout << "Connected Component by Xuhao Chen\n";
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)] [chunk_size(1024)]"
              << "[symmetrize(0/1)] [reverse(0/1)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  Graph g(argv[1], 0, 0, 0, 0, 1);
  g.print_meta_data();

  std::vector<comp_t> comp(g.V());
  // Initialize each node to a single-node self-pointing tree
  #pragma omp parallel for
  for (vidType i = 0; i < g.V(); i++) comp[i] = i;
  CCSolver(g, &comp[0]);
  CCVerifier(g, &comp[0]);
  return 0;
}
