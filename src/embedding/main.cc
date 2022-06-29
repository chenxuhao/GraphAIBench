// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <random>
/*
  Kernel: Collaborative Filtering (CF)
  Will return two latent vectors for users and items respectively.

  This algorithm solves the matrix factorization problem for recommender 
  systems using the SGD method described in [1].

  [1] Yehuda Koren, Robert Bell and Chris Volinsky, Matrix factorization
      techniques for recommender systems, IEEE Computer, 2009

  cf_omp_base  : one thread per row (vertex) using OpenMP
  cf_gpu_base  : one thread per row (vertex) using CUDA
  cf_gpu_warp  : one warp per row (vertex) using CUDA
  cf_gpu_vector: one vector per row (vertex) using CUDA

  The input to the program is a symmetrized weighted bipartite graph
  between users and items, where the weights represent the rating a
  user gives to an item. Each vertex in the graph represents either a
  user or an item. The optional arguments to the program are as follows: 
  "-K" specifies the dimension of the latent vector (default is 20), 
  "-numiter" is the number of iterations of gradient descent to run (default is 5), 
  "-step" is the step size in the algorithm (default is 0.00000035), 
  "-lambda" is the regularization parameter (default is 0.001), 
  and "-randInit" specifies that the latent vector should be initialized randomly 
  (by default every entry is initialized to 0.5).
*/

// CF parameters
float cf_epsilon = 0.1;     // convergence condition
score_t lambda = 0.001;     // regularization_factor
score_t step = 0.00000035;  // learning rate in the algorithm
int max_iters = 5;          // maximum number of iterations

void SGDSolver(BipartiteGraph &g, latent_t *latents, int *ordering);
void SGDVerifier(BipartiteGraph &g, latent_t *latents, int *ordering);

void Initialize(std::vector<latent_t> lv) {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(0, 0.1);
  for (size_t i = 0; i < lv.size(); ++i) {
    lv[i] = dist(rng);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <graph> [lambda(0.05)] [step(0.003)] [max_iter(5)] [epsilon(0.00000035)] [K(20)]\n", argv[0]);
    exit(1);
  }
  if (argc > 2) lambda = atof(argv[2]);
  if (argc > 3) step = atof(argv[3]);
  if (argc > 4) max_iters = atof(argv[4]);
  if (argc > 5) cf_epsilon = atof(argv[5]);
  //if (argc > 6) K = atoi(argv[6]);
  BipartiteGraph g(argv[1], 0, 1, 0, 1, 0, 1); // no-dag; directed; no-vlabel; elabel; no-reverse; bipartite
  auto nv = g.V();
  auto num_users = g.V(0);
  g.print_meta_data();

  std::vector<latent_t> latents_init(size_t(nv) * K);
  Initialize(latents_init);
  std::vector<latent_t> latents = latents_init;
  std::vector<int> ordering(num_users, 0);
  std::cout << "Shuffling users...\n";
  for (vidType i = 0; i < num_users; i ++) ordering[i] = i;
  std::random_shuffle(ordering.begin(), ordering.end());
  SGDSolver(g, &latents[0], &ordering[0]);
  SGDVerifier(g, &latents_init[0], &ordering[0]);
  return 0;
}
