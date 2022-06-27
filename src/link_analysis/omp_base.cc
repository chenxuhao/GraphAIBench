// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void PRSolver(Graph &g, score_t *scores) {
  if (!g.has_reverse_graph()) {
    std::cout << "This algorithm requires the reverse graph constructed for directed graph\n";
    std::cout << "Please set reverse to 1 in the command line\n";
    exit(1);
  }
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP PangeRank (" << num_threads << " threads)\n";
  auto nv = g.V();
  const score_t base_score = (1.0f - kDamp) / nv;
  score_t *outgoing_contrib = (score_t *) malloc(nv * sizeof(score_t));
  int iter = 0;
  Timer t;
  t.Start();
  for (; iter < MAX_ITER; iter ++) {
    double error = 0;
    #pragma omp parallel for
    for (int n = 0; n < nv; n ++)
      outgoing_contrib[n] = scores[n] / g.get_degree(n);
    #pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
    for (int dst = 0; dst < nv; dst ++) {
      score_t incoming_total = 0;
      for (auto src : g.in_neigh(dst))
        incoming_total += outgoing_contrib[src];
      score_t old_score = scores[dst];
      scores[dst] = base_score + kDamp * incoming_total;
      error += fabs(scores[dst] - old_score);
    }   
    printf(" %2d    %lf\n", iter+1, error);
    if (error < EPSILON) break;
  }
  t.Stop();
  std::cout << "iterations = " << iter+1 << ".\n";
  std::cout << "runtime [omp_pull_base] = " << t.Seconds() << " sec\n";
  return;
}

