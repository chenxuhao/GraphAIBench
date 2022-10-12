// Copyright 2020
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void ColorSolver(Graph &g, int *colors) {
  Timer t;
  t.Start();
  int max_color = 0;
  std::vector<int> mark(g.V(), -1);
  std::cout << "Serial vertex coloring\n";
  for (vidType u = 0; u < g.V(); u++) {
    for (auto v : g.N(u))
      mark[colors[v]] = u;
    int vertex_color = 0;
    while (vertex_color < max_color && mark[vertex_color] == u)
      vertex_color++;
    if (vertex_color == max_color)
      max_color++;
    colors[u] = vertex_color;
  }
  t.Stop();
  std::cout << "runtime [serial] = " << t.Seconds() << " sec\n";
}

