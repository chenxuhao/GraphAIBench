// Copyright 2020 MIT
// Author: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "sliding_queue.h"

void first_fit(Graph &g, SlidingQueue<vidType> &wl, int *colors) {
  #pragma omp parallel for
  for (auto iter = wl.begin(); iter < wl.end(); iter++) {
    auto u = *iter;
    vidType forbiddenColors[MAX_COLOR];
    for (int i = 0; i < MAX_COLOR; i++)
      forbiddenColors[i] = g.V() + 1;
    for (auto v : g.N(u))
      forbiddenColors[colors[v]] = u;
    int vertex_color = 0;
    while (vertex_color < MAX_COLOR && forbiddenColors[vertex_color] == u)
      vertex_color++;
    assert(vertex_color < MAX_COLOR);
    colors[u] = vertex_color;
  }
}

void conflict_resolve(Graph &g, SlidingQueue<vidType> &inwl, int *colors) {
  #pragma omp parallel
  {
  QueueBuffer<vidType> outwl(inwl);
  #pragma omp for
  for (auto iter = inwl.begin(); iter < inwl.end(); iter++) {
    auto src = *iter;
    for (auto dst : g.N(src)) {
      if (src < dst && colors[src] == colors[dst]) {
        outwl.push_back(src);
        break;
      }
    }
  }
  outwl.flush();
  }
}

void ColorSolver(Graph &g, int *colors) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP vertex coloring (" << num_threads << " threads) ...\n";
  SlidingQueue<vidType> queue(g.E());
  for (vidType j = 0; j < g.V(); j++) queue.push_back(j);
  queue.slide_window();

  Timer t;
  t.Start();
  int iter = 0;
  while (!queue.empty()) {
    ++ iter;
    first_fit(g, queue, colors);
    conflict_resolve(g, queue, colors);
    queue.slide_window();
  }
  t.Stop();
  std::cout << "runtime [omp_base] = " << t.Seconds() << " sec\n";
}

