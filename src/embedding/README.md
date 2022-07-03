Graph Emebdding, a shallow approach for graph machine learning or graph representation learning.

Stochastic Gradient Descent (SGD) based parallel algorithm for graph representation learning.

## Collaborative Filtering (Matrix Factorization) ##

The algorithm is described in the paper "GraphMat: High performance graph analytics made productive", VLDB 2015
(https://github.com/narayanan2004/GraphMat/blob/master/src/SGD.cpp)

Kernel: Collaborative Filtering (CF)
Will return latent vectors for users and items respectively.

This algorithm solves the matrix factorization problem for recommender 
systems using the SGD method described in [1].

[1] Yehuda Koren, Robert Bell and Chris Volinsky, Matrix factorization
    techniques for recommender systems, IEEE Computer, 2009

* cf_omp_base  : one thread per row (vertex) using OpenMP
* cf_gpu_base  : one thread per row (vertex) using CUDA
* cf_gpu_warp  : one warp per row (vertex) using CUDA
* cf_gpu_vector: one vector per row (vertex) using CUDA

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

## node2vec ##

## DeepWalk ##

