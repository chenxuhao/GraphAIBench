#include "graph.h"
#include <cmath>
using namespace std;

int main(int argc, char* argv[]) {
  // Graph g;
  std::string in_prefix = argv[1];
  std::string out_prefix = argv[2];
  std::string scheme = "streamvbyte";
  bool permutated = false;
  bool compress_graph = false;
  int n_idx = 0;
  int c;
  int idx = 0;
  vidType transit = 0;
  while ((c = getopt(argc, argv, "cs:t:i:")) != -1) {
    switch (c) {
      case 'c':
        compress_graph = true;
        break;
      case 's':
        n_idx = atoi(optarg);
        break;
      case 't':
        transit = (vidType)atoi(optarg);
        break;
      case 'i':
        idx = atoi(optarg);
        break;
      default:
        abort();
    }
  }
  Graph g(in_prefix, 0, 0, 0, 0, 0);
 // std::cout << "deg " << g.get_degree(transit) << "\nneighbor " << g.N(transit, idx) << std::endl;
  std::cout << "deg " << g.get_degree(81023274) << "\nneighbor " << g.N(81023274, 51) << std::endl;
  std::cout << "deg " << g.get_degree(280280417) << "\nneighbor " << g.N(280280417, 50) << std::endl;
  std::cout << "deg " << g.get_degree(626) << "\nneighbor " << g.N(626, 49880) << std::endl;
  std::cout << "deg " << g.get_degree(938454) << "\nneighbor " << g.N(938454, 1138) << std::endl; 
}
