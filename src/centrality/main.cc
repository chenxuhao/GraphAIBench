// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "common.h"
#include "graph.h"
/*
GARDENIA Benchmark Suite
Kernel: Betweenness Centrality (BC)
Author: Xuhao Chen

Will return array of approx betweenness centrality scores for each vertex

This BC implementation makes use of the Brandes [1] algorithm with
implementation optimizations from Madduri et al. [2]. It is only an approximate
because it does not compute the paths from every start vertex, but only a small
subset of them. Additionally, the scores are normalized to the range [0,1].

As an optimization to save memory, this implementation uses a Bitmap to hold
succ (list of successors) found during the BFS phase that are used in the back-
propagation phase.

[1] Ulrik Brandes. "A faster algorithm for betweenness centrality." Journal of
    Mathematical Sociology, 25(2):163–177, 2001.

[2] Kamesh Madduri, David Ediger, Karl Jiang, David A Bader, and Daniel
	Chavarria-Miranda. "A faster parallel algorithm and efficient multithreaded
	implementations for evaluating betweenness centrality on massive datasets."
	International Symposium on Parallel & Distributed Processing (IPDPS), 2009.

bc_omp: OpenMP implementation, one thread per vertex
bc_topo_base: topology-driven GPU implementation, one thread per vertex using CUDA
bc_topo_lb: topology-driven GPU implementation, one thread per edge using CUDA
bc_linear_base: data-driven GPU implementation, one thread per vertex using CUDA
bc_linear_lb: data-driven GPU implementation, one thread per edge using CUDA
*/

void BCSolver(Graph &g, int source, ScoreT *scores);
void BCVerifier(Graph &g, int source, int num_iters, ScoreT *scores_to_test);

int main(int argc, char *argv[]) {
  std::cout << "Betweenness Centrality by Xuhao Chen\n";
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <filetype> <graph-prefix> "
              << "[symmetrize(0/1)] [reverse(0/1)] [source_id(0)]\n";
    std::cout << "Example: " << argv[0] << " mtx web-Google\n";
    exit(1);
  }
  bool symmetrize = false;
  bool need_reverse = false;
  if (argc > 3) symmetrize = atoi(argv[3]);
  if (argc > 4) need_reverse = atoi(argv[4]);
  Graph g(argv[2], argv[1], symmetrize, need_reverse);
  int source = 0;
  if (argc == 6) source = atoi(argv[5]);
  auto m = g.V();
  std::vector<ScoreT> scores(m, 0);
  BCSolver(g, source, &scores[0]);
  BCVerifier(g, source, 1, &scores[0]);
  return 0;
}
