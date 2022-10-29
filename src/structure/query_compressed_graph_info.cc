#include "graph.h"

int main(int argc,char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph\n";
    abort();
  }
  Graph g;
  g.load_compressed_graph(argv[1]);
  g.print_meta_data();
 
  int v = atoi(argv[2]);
  //auto deg = g.get_degree(v);
  auto adj = g.N_compressed(v, true);
  std::cout << "vertex " << v << " degree " << adj.size() << "\n";
  for (size_t i = 0; i < (adj.size()>10?10:adj.size()); i++) {
    std::cout << "neighbor[" << i << "]=" << adj[i] << "\n";
  }
  auto adj_itv = g.get_interval_neighbors(v);
  std::cout << "vertex " << v << " num_interval_neighbors " << adj_itv.size() << "\n";
  return 0;
}
