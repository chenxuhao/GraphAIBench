// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

inline score_t rmse(int nv, int ne, score_t *errors);

void SGDVerifier(BipartiteGraph &g, score_t *ratings, latent_t *user_lv, latent_t *item_lv, int *ordering) {
	std::cout << "Verifying...\n";
#ifdef COMPUTE_ERROR
	score_t *squared_errors = (score_t *)malloc(m * sizeof(score_t));
	score_t total_error = 0.0;
#endif
  auto num_users = g.V(0);
  //auto num_items = g.V(1);

	int iter = 0;
	Timer t;
	t.Start();
	do {
		iter ++;
#ifdef COMPUTE_ERROR
		for (int i = 0; i < num_users; i ++) squared_errors[i] = 0;
#endif

		for(int i = 0; i < num_users; i ++) {
			//int src = ordering[i];
			int src = i;
      auto offset = g.edge_begin(src);
			for (auto dst : g.N(src)) {
				score_t estimate = 0;
				for (int i = 0; i < K; i++) {
					estimate += user_lv[src*K+i] * item_lv[dst*K+i];
				}
				score_t delta = ratings[offset++] - estimate;
#ifdef COMPUTE_ERROR
				squared_errors[src] += delta * delta;
#endif
				for (int i = 0; i < K; i++) {
					auto p_s = user_lv[src*K+i];
					auto p_d = item_lv[dst*K+i];
					user_lv[src*K+i] += step * (-lambda * p_s + p_d * delta);
					item_lv[dst*K+i] += step * (-lambda * p_d + p_s * delta);
				}
			}
		}
#ifdef COMPUTE_ERROR
		total_error = rmse(g.V(), g.E(), squared_errors);
		printf("Iteration %d: RMSE error = %f\n", iter, total_error);
		if (total_error < cf_epsilon) break;
#endif
	} while (iter < max_iters);
	t.Stop();
	printf("\titerations = %d.\n", iter);
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
#ifdef COMPUTE_ERROR
	free(squared_errors);
#endif
	return;
}

