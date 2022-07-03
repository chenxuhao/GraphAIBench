// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void BFSSolver(Graph &g, int source, vidType *dist);
void BFSVerifier(Graph &g, int source, vidType *depth_to_test);
void SSSPSolver(Graph &g, int source, elabel_t *dist, int delta);
void SSSPVerifier(Graph &g, int source, elabel_t *dist);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>"
      << " [source_id(0)] [delta(-1)]\n";
    //<< " [num_gpu(1)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  int source = 0;
  if (argc > 2) source = atoi(argv[2]);
  std::cout << "Source vertex: " << source << "\n";
  int delta = -1;
  if (argc > 3) delta = atoi(argv[3]);
  if (delta == -1) {
    std::cout << "Breadth-first Search\n";
    Graph g(argv[1], 0, 1);
    g.print_meta_data();
    std::vector<vidType> distances(g.V(), MYINFINITY);
    BFSSolver(g, source, &distances[0]);
    BFSVerifier(g, source, &distances[0]);
  } else {
    assert(delta > 0);
    std::cout << "Single-source Shortest Paths\n";
    Graph g(argv[1], 0, 1, 0, 1);
    g.print_meta_data();
    std::vector<elabel_t> distances(g.V(), kDistInf);
    SSSPSolver(g, source, &distances[0], delta);
    SSSPVerifier(g, source, &distances[0]);
  }
  return 0;
}
