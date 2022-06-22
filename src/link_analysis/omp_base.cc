// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <stdlib.h>

void PRSolver(Graph &g, score_t *scores) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("Launching OpenMP PR solver (%d threads) ...\n", num_threads);
  auto m = g.V();
  const score_t base_score = (1.0f - kDamp) / m;
  score_t *outgoing_contrib = (score_t *) malloc(m * sizeof(score_t));
  int iter;
  Timer t;
  t.Start();
  for (iter = 0; iter < MAX_ITER; iter ++) {
    double error = 0;
    #pragma omp parallel for
    for (int n = 0; n < m; n ++)
      outgoing_contrib[n] = scores[n] / g.get_degree(n);
    #pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
    for (int dst = 0; dst < m; dst ++) {
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
  printf("\titerations = %d.\n", iter+1);
  printf("\truntime [omp_base] = %f ms.\n", t.Millisecs());
  return;
}

