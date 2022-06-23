// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void BFSSolver(Graph &g, int source, vidType *dist);
void BFSVerifier(Graph &g, int source, vidType *depth_to_test);

int main(int argc, char *argv[]) {
	std::cout << "Breadth-first Search\n";
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)] [chunk_size(1024)]"
              << "[symmetrize(0/1)] [reverse(0/1)] [source_id(0)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  bool symmetrize = false;
  bool need_reverse = false;
  if (argc > 3) symmetrize = atoi(argv[3]);
  if (argc > 4) need_reverse = atoi(argv[4]);
  Graph g(argv[1]);
  g.print_meta_data();

  int source = 0;
  if (argc == 6) source = atoi(argv[5]);
  std::vector<vidType> distances(g.V(), MYINFINITY);
  BFSSolver(g, source, &distances[0]);
  BFSVerifier(g, source, &distances[0]);
  return 0;
}
