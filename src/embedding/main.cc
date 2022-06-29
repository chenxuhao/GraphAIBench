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

void SGDSolver(BipartiteGraph &g, score_t *ratings, 
               latent_t *user_lv, latent_t *item_lv, int *ordering);
void SGDVerifier(BipartiteGraph &g, score_t *ratings, 
                 latent_t *user_lv, latent_t *item_lv, int *ordering);

#define COMPUTE_ERROR

void Initialize(std::vector<latent_t> lv) {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(0, 0.1);
  for (size_t i = 0; i < lv.size(); ++i) {
    lv[i] = dist(rng);
  }
}

// calculate RMSE
inline score_t rmse(int nv, int ne, score_t *errors) {
	score_t total_error = 0.0;
	for(int i = 0; i < nv; i ++)
		total_error += errors[i];
	total_error = sqrt(total_error/ne);
	return total_error;
}

inline score_t rmse_par(int nv, int ne, score_t *errors) {
  score_t total_error = 0.0;
  #pragma omp parallel for reduction(+ : total_error)
  for(int i = 0; i < nv; i ++) {
    total_error += errors[i];
  }
  total_error = sqrt(total_error/ne);
  return total_error;
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
  BipartiteGraph g(argv[1]);
  auto num_users = g.V(0);
  auto num_items = g.V(1);

  std::vector<latent_t> user_lv(size_t(num_users) * K);
  std::vector<latent_t> item_lv(size_t(num_items) * K);
  std::vector<latent_t> lv_u(size_t(num_users) * K);
  std::vector<latent_t> lv_i(size_t(num_items) * K);
  std::vector<score_t> ratings(g.E());
  Initialize(lv_u);
  Initialize(lv_i);
  for (size_t i = 0; i < lv_u.size(); i++) user_lv[i] = lv_u[i];
  for (size_t i = 0; i < lv_i.size(); i++) item_lv[i] = lv_i[i];
  std::vector<int> ordering(num_users, 0);
  std::cout << "Shuffling users...\n";
  for (int i = 0; i < num_users; i ++) ordering[i] = i;
  std::random_shuffle(ordering.begin(), ordering.end());
  SGDSolver(g, &ratings[0], &user_lv[0], &item_lv[0], &ordering[0]);
  SGDVerifier(g, &ratings[0], &lv_u[0], &lv_i[0], &ordering[0]);
  return 0;
}
