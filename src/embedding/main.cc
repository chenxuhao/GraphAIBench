// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <random>

// CF parameters
score_t cf_epsilon = 0.1;    // convergence condition
score_t lambda = 0.001;     // regularization_factor
score_t step = 0.00000035;  // learning rate in the algorithm
int max_iters = 5;          // maximum number of iterations

void SGDSolver(BipartiteGraph &g, std::vector<latent_t> &latents, int *ordering);
void SGDVerifier(BipartiteGraph &g, std::vector<latent_t> &latents, int *ordering);

void Initialize(std::vector<latent_t> &lv) {
  size_t nv = lv.size() / K;
  //std::cout << "nv = " << nv << "\n";
  for (size_t i = 0; i < nv; ++i) {
    std::default_random_engine rng;
    std::uniform_real_distribution<float> dist(0, 1);
    //srand(0);
    //long seed = rand();
    for (int j = 0; j < K; j++) {
      //lv[i*K+j] = 0.5; 
      lv[i*K+j] = dist(rng);
      //lv[i*K+j] = (float)(seed+hashInt((uintE)i*K+j))/(float)UINT_E_MAX;
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <graph> [lambda(0.001)] [step(0.00000035)] [max_iter(5)] [epsilon(0.1)] [K(20)]\n", argv[0]);
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
  auto num_items = g.V(1);
  g.print_meta_data();
  std::cout << "# users: " << num_users << " ; # items: " << num_items << "\n";
  std::cout << "lambda: " << lambda << " ; step: " << step << " ; max_iter: " 
            << max_iters << " ; epsilon: " << cf_epsilon << " ; K: " << K << "\n";
  //g.print_graph();

  std::vector<latent_t> latents_init(size_t(nv) * K);
  Initialize(latents_init);
  std::vector<latent_t> latents(latents_init);
  std::vector<int> ordering(num_users, 0);
  //std::cout << "Shuffling users...\n";
  //for (vidType i = 0; i < num_users; i ++) ordering[i] = i;
  std::random_shuffle(ordering.begin(), ordering.end());
  SGDSolver(g, latents, &ordering[0]);
  SGDVerifier(g, latents_init, &ordering[0]);
  return 0;
}
