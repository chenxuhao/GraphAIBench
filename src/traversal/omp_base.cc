// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"

void bfs_step(Graph &g, vidType *depth, SlidingQueue<vidType> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<vidType> lqueue(queue);
    #pragma omp for
    for (vidType *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      vidType src = *q_iter;
      for (auto dst : g.N(src)) {
        //int curr_val = parent[dst];
        int curr_val = depth[dst];
        if (curr_val == MYINFINITY) { // not visited
          //if (compare_and_swap(parent[dst], curr_val, src)) {
          if (compare_and_swap(depth[dst], curr_val, depth[src] + 1)) {
            lqueue.push_back(dst);
          }
        }
      }
    }
    lqueue.flush();
  }
}

void BFSSolver(Graph &g, int source, vidType* dist) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "Launching OpenMP BFS solver (" << num_threads << " threads) ...\n";
  VertexList depth(g.V(), MYINFINITY);
  depth[source] = 0;
  int iter = 0;
  Timer t;
  t.Start();
  SlidingQueue<vidType> queue(g.E());
  queue.push_back(source);
  queue.slide_window();
  while (!queue.empty()) {
    ++ iter;
    std::cout << "iteration=" << iter << ", frontier_size=" << queue.size() << "\n";
    bfs_step(g, depth.data(), queue);
    queue.slide_window();
  }
  t.Stop();
  std::cout << "\titerations = " << iter << "\n";
  std::cout << "\truntime [omp_base] = " << t.Seconds() << "sec\n";
  #pragma omp parallel for
  for (int i = 0; i < g.V(); i ++)
    dist[i] = depth[i];
}

void SSSPSolver(Graph &g, int source, elabel_t *dist, int delta) {}
