// Copyright 2022
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"

void PBFS(Graph &g, int source, std::vector<int> &path_counts, 
          std::vector<int> &depths, Bitmap &succ,
          std::vector<SlidingQueue<vidType>::iterator> &depth_index, 
          SlidingQueue<vidType> &queue) {
  depths[source] = 0;
  path_counts[source] = 1;
  queue.push_back(source);
  depth_index.push_back(queue.begin());
  queue.slide_window();
  #pragma omp parallel
  {
    int depth = 0;
    QueueBuffer<vidType> lqueue(queue);
    while (!queue.empty()) {
      #pragma omp single
      depth_index.push_back(queue.begin());
      depth++;
      #pragma omp for schedule(dynamic, 64)
      for (vidType *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
        auto src = *q_iter;
        auto offset = g.edge_begin(src);
        for (auto dst : g.N(src)) {
          if (depths[dst] == -1  && (compare_and_swap(depths[dst], -1, depth))) {
            lqueue.push_back(dst);
          }
          if (depths[dst] == depth) {
            succ.set_bit_atomic(offset);
            fetch_and_add(path_counts[dst], path_counts[src]);
          }
          offset ++;
        }
      }
      lqueue.flush();
      #pragma omp barrier
      #pragma omp single
      queue.slide_window();
    }
  }
  depth_index.push_back(queue.begin());
}

void BCSolver(Graph &g, int source, score_t *scores) {
  auto m = g.V();
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP BC (" << num_threads << " threads)\n";
  int num_iters = 1;
  Bitmap succ(g.E());
  vector<SlidingQueue<vidType>::iterator> depth_index;

  Timer t;
  t.Start();
  int depth = 0;
  SlidingQueue<vidType> queue(m);
  for (int iter = 0; iter < num_iters; iter++) {
    vector<int> path_counts(m, 0);
    vector<int> depths(m, -1);
    depth_index.resize(0);
    queue.reset();
    succ.reset();
    PBFS(g, source, path_counts, depths, succ, depth_index, queue);
    vector<score_t> deltas(m, 0);
    for (int d = depth_index.size()-2; d >= 0; d --) {
      depth ++;
      auto nitems = depth_index[d+1] - depth_index[d];
      printf("Reverse: depth=%d, frontier_size=%ld\n", d, nitems);
      #pragma omp parallel for schedule(dynamic, 64)
      for (vidType *it = depth_index[d]; it < depth_index[d+1]; it++) {
        auto src = *it;
        score_t delta_src = 0;
        auto offset = g.edge_begin(src);
        for (auto dst : g.N(src)) {
          if (succ.get_bit(offset)) {
            delta_src += static_cast<score_t>(path_counts[src]) /
              static_cast<score_t>(path_counts[dst]) * (1 + deltas[dst]);
          }
          offset ++;
        }
        deltas[src] = delta_src;
        scores[src] += delta_src;
      }
    }
  }
  // Normalize scores
  score_t biggest_score = 0;
  #pragma omp parallel for reduction(max : biggest_score)
  for (int n = 0; n < m; n ++)
    biggest_score = max(biggest_score, scores[n]);
  #pragma omp parallel for
  for (int n = 0; n < m; n ++)
    scores[n] = scores[n] / biggest_score;
  t.Stop();

  std::cout << "iterations = " << depth << ".\n";
  std::cout << "runtime [bc_gpu_base] = " << t.Seconds() << " sec\n";
  return;
}
