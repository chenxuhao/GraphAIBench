#include "graph_partition.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "test graph partitioning\n";
  Graph g(argv[1]);
  int n_devices = 1;
  int chunk_size = 1024;
  if (argc > 2) n_devices = atoi(argv[2]);
  if (argc > 3) chunk_size = atoi(argv[3]);
  g.print_meta_data();
  uint64_t total = 0;
  PartitionedGraph(g, n_devices);
  //std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}

