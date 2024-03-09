#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "sampling_utils.h"
#include "samplegraph.h"
using namespace std;


int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <graph>"
              << "[num_gpu(1)] [chunk_size(1024)]\n";
    cout << "Example: " << argv[0] << " ../inputs/cora/graph\n";
    exit(1);
  }

  // create graph and retrieve node/edge data
  Graph g(argv[1], 0, 0, 0, 0, 0);
  return 0;
};
