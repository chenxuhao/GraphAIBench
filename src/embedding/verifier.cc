// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

// calculate RMSE
inline score_t rmse(int nv, int ne, score_t *errors) {
	score_t total_error = 0.0;
	for(int i = 0; i < nv; i ++)
		total_error += errors[i];
	total_error = sqrt(total_error/ne);
	return total_error;
}

void SGDVerifier(BipartiteGraph &g, latent_t *latents, int *ordering) {
  std::cout << "Verifying...\n";
  auto num_users = g.V(0);
  //auto num_items = g.V(1);
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
    std::fill(squared_errors.begin(), squared_errors.end(), 0.0);
    //for (int i = 0; i < num_users; i ++) squared_errors[i] = 0.0;
#endif

    for(int i = 0; i < num_users; i ++) {
      //int src = ordering[i];
      int src = i;
      auto offset = g.edge_begin(src);
      latent_t* user_lv = &latents[src*K];
      for (auto dst : g.N(src)) {
        latent_t* item_lv = &latents[dst*K];
        score_t estimate = 0;
        for (int i = 0; i < K; i++) {
          estimate += user_lv[i] * item_lv[i];
        }
        score_t rating = g.getEdgeData(offset++);
        score_t delta = rating - estimate;
#ifdef COMPUTE_ERROR
        squared_errors[src] += delta * delta;
#endif
        for (int i = 0; i < K; i++) {
          auto p_s = user_lv[i];
          auto p_d = item_lv[i];
          user_lv[i] += step * (-lambda * p_s + p_d * delta);
          item_lv[i] += step * (-lambda * p_d + p_s * delta);
        }
      }
    }
#ifdef COMPUTE_ERROR
    total_error = rmse(num_users, g.E(), &squared_errors[0]);
    printf("Iteration %d: RMSE error = %f\n", iter, total_error);
    if (total_error < cf_epsilon) break;
#endif
  } while (iter < max_iters);
  t.Stop();
  std::cout << "iterations = " << iter << ".\n";
  std::cout << "runtime [verify] = " << t.Seconds() << " sec\n";
  return;
}

