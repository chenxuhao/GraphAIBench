// Copyright 2022
// Author: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

inline score_t rmse_par(int nv, int ne, score_t *errors) {
  score_t total_error = 0.0;
  #pragma omp parallel for reduction(+ : total_error)
  for(int i = 0; i < nv; i ++) {
    total_error += errors[i];
  }
  total_error = sqrt(total_error/ne);
  return total_error;
}

void SGDSolver(BipartiteGraph &g, latent_t *latents, int * ordering) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP CF (" << num_threads << " threads)\n";
  auto num_users = g.V(0);
  auto num_items = g.V(1);

  std::vector<score_t> user_error(num_users*K, 0);
  std::vector<score_t> item_error(num_items*K, 0);
  #ifdef COMPUTE_ERROR
  std::vector<score_t> squared_errors(num_users, 0.0);
  score_t total_error = 0.0;
  #endif

  int iter = 0;
  Timer t;
  t.Start();
  do {
    iter ++;
    #ifdef COMPUTE_ERROR
    for (int i = 0; i < num_users; i ++) squared_errors[i] = 0;
    #endif
    #pragma omp parallel for schedule(dynamic, 64)
    for (int u = 0; u < num_users; u ++) {
      latent_t *user_latent = &latents[K*u];
      //latent_t *uerr = &user_error[K*u];
      auto offset = g.edge_begin(u);
      for (auto v : g.N(u)) {
        latent_t *item_latent = &latents[K*v];
        //latent_t *ierr = &item_error[K*v];
        score_t estimate = 0;
        for (int i = 0; i < K; i++) {
          estimate += user_latent[i] * item_latent[i];
        }
        score_t rating = g.getEdgeData(offset++);
        score_t delta = rating - estimate;
        #ifdef COMPUTE_ERROR
        squared_errors[u] += delta * delta;
        #endif
        for (int i = 0; i < K; i++) {
          auto p_d = user_latent[i];
          auto p_s = item_latent[i];
          user_latent[i] += step * (-lambda * p_d + p_s * delta);
          item_latent[i] += step * (-lambda * p_s + p_d * delta);
          //uerr[i] += item_latent[i] * delta;
          //ierr[i] += user_latent[i] * delta;
        }
      }
    }
    #ifdef COMPUTE_ERROR
    total_error = rmse_par(num_users, g.E(), &squared_errors[0]);
    printf("Iteration %d: RMSE error = %f\n", iter, total_error);
    if (total_error < cf_epsilon) break;
    #endif
  } while (iter < max_iters);
  t.Stop();
  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [cf_omp_base] = " << t.Seconds() << " sec\n";
  return;
}
