## Betweenness Centrality (BC) ##

Will return array of approximate betweenness centrality scores for each vertex

This BC implementation makes use of the Brandes [1] algorithm with
implementation optimizations from Madduri et al. [2]. It is only an approximate
because it does not compute the paths from every start vertex, but only a small
subset of them. Additionally, the scores are normalized to the range [0,1].

As an optimization to save memory, this implementation uses a Bitmap to hold
succ (list of successors) found during the BFS phase that are used in the back-
propagation phase.

[1] Ulrik Brandes. "A faster algorithm for betweenness centrality." Journal of
    Mathematical Sociology, 25(2):163--177, 2001.

[2] Kamesh Madduri, David Ediger, Karl Jiang, David A Bader, and Daniel
	Chavarria-Miranda. "A faster parallel algorithm and efficient multithreaded
	implementations for evaluating betweenness centrality on massive datasets."
	International Symposium on Parallel & Distributed Processing (IPDPS), 2009.

```
bc_omp_base: OpenMP implementation, one thread per vertex
bc_topo_base: topology-driven GPU implementation, one thread per vertex
bc_topo_twc: topology-driven GPU implementation, one thread per edge using TWC load balancing
bc_gpu_base: data-driven GPU implementation, one thread per vertex
bc_gpu_twc: data-driven GPU implementation, one thread per edge using TWC load balancing
```
