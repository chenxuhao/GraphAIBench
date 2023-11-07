#include "graph.h"
#include <vector>
#include <iostream>

int* node2vec(Graph& g, std::vector<int> sources, int hops);

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>"
              << "[num_gpu(1)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/cora/graph\n";
    exit(1);
  }
  Graph g(argv[1], 0 , 1, 0, 0, 1);
  g.print_meta_data();

  
  const int hops = 16;
  const int numSources = 25;

  std::vector <int> sources;
  for(int i = 0; i < numSources; i++) {
    sources.push_back(2*i);
  }

  int* paths = node2vec(g, sources, hops);

  for(int i = 0; i < numSources; i++) {
    std::cout << "source " << sources[i] << ": ";
    for(int j = 0; j < hops; j++) {
      int node = paths[i * hops + j];
      std::cout << node << " ";
    }
    std::cout << std::endl;
  }
  free(paths);
  return 0;
}

