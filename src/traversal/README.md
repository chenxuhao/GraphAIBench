# Traversal
Graph traversal algorithms: Breadth-first Search (BFS) and Single Source Shortest Path (SSSP).

## SSSP

Kernel: Single-source Shortest Paths (SSSP)
Returns array of distances for all vertices from given source vertex

Two parallel algorithms are often used for SSSP: 
(1) Bellman Ford;
(2) Delta Stepping [1].

This SSSP implementation makes use of the δ-stepping algorithm [1].
The type used for weights and distances (WeightT) is typedefined in benchmark.h. 
The delta parameter (-d) should be set for each input graph.

The bins of width delta are actually all thread-local and of type std::vector
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies their selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and it now appears in a lower bin. We find ignoring vertices if
their current distance is less than the min distance for the bin to remove enough 
redundant work that this is faster than removing the vertex from older bins.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest
	path algorithm." Journal of Algorithms, 49(1):114--152, 2003.

[2] A. Davidson, S. Baxter, M. Garland, and J. D. Owens, "Work-efficient
	parallel gpu methods for single-source shortest paths", in Proceedings
	of the IEEE 28th International Parallel and Distributed Processing
	Symposium (IPDPS), pp. 349--359, May 2014

* sssp_omp_base: OpenMP implementation using delta-stepping algorithm, one thread per vertex
* sssp_topo_base: topology-driven GPU implementation, one thread per vertex using CUDA
* sssp_topo_twc: topology-driven GPU implementation, one thread per edge using CUDA
* sssp_gpu_base: data-driven GPU implementation, one thread per vertex using CUDA
* sssp_gpu_twc: data-driven GPU using TWC load balancing, one thread per edge using CUDA
* sssp_gpu_dstep: data-driven GPU using delta stepping algorithm, one thread per edge using CUDA

## BFS

Kernel: Breadth-First Search (BFS)

Will return distance (or parent) array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.

* bfs_omp_base: naive OpenMP implementation using sliding queue, one thread per vertex
* bfs_omp_direction: Beamer's OpenMP implementation using the Direction Optimization, one thread per vertex
* bfs_topo_base: topology-driven GPU implementation, one thread per vertex using CUDA
* bfs_gpu_base: data-driven GPU implementation, one thread per vertex using CUDA
* bfs_gpu_twc: data-driven GPU using TWC load balancing, one thread per edge using CUDA
