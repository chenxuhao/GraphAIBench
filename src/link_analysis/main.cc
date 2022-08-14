// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
/*
Kernel: PageRank (PR)
Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. This is done
to ease comparisons to other implementations (often use same algorithm), but
it is not necesarily the fastest way to implement it. It does perform the
updates in the pull direction to remove the need for atomics.

pr_omp_base: OpenMP implementation, one thread per vertex
pr_gpu_base: topology-driven GPU implementation using pull approach, one thread per vertex using CUDA
pr_gpu_push: topology-driven GPU implementation using push approach, one thread per edge using CUDA
*/

void PRSolver(Graph &g, score_t *scores);
void PRVerifier(Graph &g, score_t *scores, double target_error);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)]"
      << " [symmetrize(0/1)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  int reverse = 1;
  if (argc > 2) reverse = atoi(argv[2]);
  Graph g(argv[1], 0 , 1, 0, 0, reverse);
  g.print_meta_data();

  const score_t init_score = 1.0f / g.V();
  std::cout << "PageRank: initial score = " << init_score << "\n";
  std::vector<score_t> scores(g.V(), init_score);
  PRSolver(g, &scores[0]);
  PRVerifier(g, &scores[0], EPSILON);
  return 0;
}

