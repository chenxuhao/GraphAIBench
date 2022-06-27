// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include <random>
/*
GARDENIA Benchmark Suite
Kernel: Stochastic Gradient Descent (SGD)
Author: Xuhao Chen 

Will return two latent vectors for users and items respectively.

This algorithm solves the matrix factorization problem for recommender 
systems using the SGD method described in [1].

[1] Yehuda Koren, Robert Bell and Chris Volinsky, Matrix factorization
	techniques for recommender systems,” IEEE Computer, 2009

s_omp : one thread per row (vertex) using OpenMP
sgd_base: one thread per row (vertex) using CUDA
sgd_warp: one warp per row (vertex) using CUDA
sgd_vector: one vector per row (vertex) using CUDA
*
*/
#define K (20)               // dimension of the latent vector (number of features)
static ScoreT lambda = 0.001; // regularization_factor
static ScoreT step = 0.00000035;  // learning rate in the algorithm
static float epsilon = 0.1;  // convergence condition
static int max_iters = 3;    // maximum number of iterations

void SGDSolver(int m, int n, int nnz, IndexT *row_offsets, IndexT *column_indices, ScoreT *rating, 
	LatentT *user_lv, LatentT *item_lv, int *ordering);
void SGDVerifier(int m, int n, int nnz, IndexT *row_offsets, IndexT *column_indices, ScoreT *rating, 
	LatentT *user_lv, LatentT *item_lv, int *ordering);

#define COMPUTE_ERROR

void Initialize(int len, LatentT *lv) {
	std::default_random_engine rng;
	std::uniform_real_distribution<float> dist(0, 0.1);
	for (int i = 0; i < len; ++i) {
		for (int j = 0; j < K; ++j) {
			lv[i*K+j] = dist(rng);
		}
	}
	/*
	//srand(0);
	for (int i = 0; i < m; i++) {
		unsigned r = i;
		for (int j = 0; j < K; j++)
			init_user_lv[i*K+j] = ((LatentT)rand_r(&r)/(LatentT)RAND_MAX);
	}
	for (int i = 0; i < n; i++) {
		unsigned r = i + m;
		for (int j = 0; j < K; j++)
			init_item_lv[i*K+j] = ((LatentT)rand_r(&r)/(LatentT)RAND_MAX);
	}
	*/
}

int main(int argc, char *argv[]) {
	printf("Stochastic Gradient Descent by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [lambda(0.05)] [step(0.003)] [max_iter(1)]\n", argv[0]);
		exit(1);
	}
	if (argc > 2) lambda = atof(argv[2]);
	if (argc > 3) step = atof(argv[3]);
	if (argc > 4) max_iters = atof(argv[4]);
	if (argc > 5) epsilon = atof(argv[5]);
	int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false, false, false, false, false);
	printf("num_users=%d, num_items=%d\n", m, n);
	printf("regularization_factor=%f, learning_rate=%f\n", lambda, step);
	printf("max_iters=%d, epsilon=%f\n", max_iters, epsilon);

	LatentT *h_user_lv = (LatentT *)malloc(m * K * sizeof(LatentT));
	LatentT *h_item_lv = (LatentT *)malloc(n * K * sizeof(LatentT));
	LatentT *lv_u = (LatentT *)malloc(m * K * sizeof(LatentT));
	LatentT *lv_i = (LatentT *)malloc(n * K * sizeof(LatentT));
	ScoreT *h_rating = (ScoreT *) malloc(nnz * sizeof(ScoreT));

	Initialize(m, lv_u);
	Initialize(n, lv_i);
	for (int i = 0; i < m * K; i++) h_user_lv[i] = lv_u[i];
	for (int i = 0; i < n * K; i++) h_item_lv[i] = lv_i[i];
	for (int i = 0; i < nnz; i++) h_rating[i] = (ScoreT)h_weight[i];
	int *ordering = NULL;
/*
	printf("Shuffling users...\n");
	ordering = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i ++) ordering[i] = i;
	std::random_shuffle(ordering, ordering+m);
//*/
	SGDSolver(m, n, nnz, h_row_offsets, h_column_indices, h_rating, h_user_lv, h_item_lv, ordering);
	SGDVerifier(m, n, nnz, h_row_offsets, h_column_indices, h_rating, lv_u, lv_i, ordering);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	free(h_weight);
	free(h_user_lv);
	free(h_item_lv);
	free(lv_u);
	free(lv_i);
	free(h_rating);
	return 0;
}
