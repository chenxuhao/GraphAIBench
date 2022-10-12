// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void ColorSolver(Graph &g, int *colors);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)] [chunk_size(1024)] [adj_sorted(1)]\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "Vertex Coloring\n";
  Graph g(argv[1]);
  g.print_meta_data();

  std::vector<int> colors(g.V(), MAX_COLOR);
  ColorSolver(g, &colors[0]);

  int max_color = 0;
  #pragma omp parallel for reduction(max : max_color)
  for (vidType n = 0; n < g.V(); n ++)
    max_color = max(max_color, colors[n]);
  auto num_colors = max_color+1;
  std::cout << "total_num_colors = " << num_colors << "\n";
  return 0;
}

