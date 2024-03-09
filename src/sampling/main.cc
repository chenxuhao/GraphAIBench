#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "sampling_utils.h"
using namespace std;

int sample_alg(Graph &g, vector<vidType> &all_transits, int n_samples, int n_threads);

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>"
              << "[num_gpu(1)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/cora/graph\n";
    exit(1);
  }

  int n_samples = argc >= 3 ? atoi(argv[2]) : num_samples();
  int n_threads = argc >= 4 ? atoi(argv[3]) : 1;
  // create graph and retrieve node/edge data
  Graph g(argv[1], 0, 0, 0, 0, 0);

  vector<vidType> inits = get_initial_transits(sample_size(-1) * n_samples, g.V());
  int step_count = sample_size(-1) * n_samples;
  int total_count = step_count;
  for (int step = 0; step < steps(); step++) {
    step_count *= sample_size(step);
    total_count += step_count;
  }
  vector<vidType> transits(total_count, 0);
  for (int i = 0; i < inits.size(); i++) {
    transits[i] = inits[i];
  };

  sample_alg(g, transits, n_samples, n_threads);
  // print the output for checking purposes, only do for small sample sizes
  // std::cout << "results\n";
  // int _size = sample_size(-1) * n_samples;
  // int p_size = 0;
  // for (int step = 0; step <= steps(); step++) {
  //   std::cout << "\n";
  //   for (int i = 0; i < _size; i++) {
  //       std::cout << transits[i + p_size] << " ";
  //       // cout << i + p_size << " ";
  //   }
  //   p_size += _size;
  //   _size *= sample_size(step);
  // }

  return 0;
};
