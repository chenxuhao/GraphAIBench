#include "graph.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>"
              << "[num_gpu(1)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/cora/graph\n";
    exit(1);
  }
  Graph g(argv[1], 0 , 1, 0, 0, 1);
  g.print_meta_data();

  return 0;
}

