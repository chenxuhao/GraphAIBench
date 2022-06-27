PageRank
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Author: Xuhao Chen <cxh@mit.edu>

This program implements the PageRank algorithms. 
It iteratively updates vertex scores using neighborhood propagation until convergence, i.e., total score change < epsilon. 
The program returns pagerank scores for all vertices.

This PageRank implementation uses the traditional iterative approach. 
This is done to ease comparisons to other implementations (often use the same algorithm), 
but it is not necesarily the fastest way to implement it. 
It does perform the updates in the pull direction to remove the need for atomics.

INPUT GRAPH FORMATS
--------------------------------------------------------------------------------

The graph loading infrastructure understands the following formats:

+ `graph.meta.txt` text file specifying the meta information of the graph, including the number of vertices and edges; number of bytes for vertex IDs, edge IDs, vertex labels, and edge labels; maximum degree; feature vector length; distinct vertex label and edge label count.

+ `graph.vertex.bin` binary file containing the row pointers, with data type of edge IDs.

+ `graph.edge.bin` binary file containing the column indices, with data type of vertex IDs.

+ `graph.vlabel.bin` binary file containing the vertex labels (only needed for vertex labeled graphs)

+ `graph.elabel.bin` binary file containing the edge labels (only needed for edge labeled graphs)


Example graphs are in inputs/citeseer and inputs/mico

More datasets are available [here](https://www.dropbox.com/sh/i1jq1uwtkcd2qo0/AADJck_u3kx7FeSR5BvdrkqYa?dl=0).

BUILD
--------------------------------------------------------------------------------

1. Run make at this directory

2. Or run make at the top-level directory

The following binaries will be generated in ../../bin/

  - pr_omp_base : one thread per vertex using OpenMP
  - pr_gpu_base : one thread per vertex using CUDA
  - pr_gpu_warp : one warp per vertex using CUDA

RUN
--------------------------------------------------------------------------------

The following are example command lines:

`$ ../../bin/pr_omp_base ../../inputs/mico/graph`

OUTPUT
```
OpenMP PangeRank (8 threads)
  1    0.368150
  2    0.121518
  3    0.053159
  4    0.029254
  5    0.018512
  6    0.012585
  7    0.009073
  8    0.006688
  9    0.005065
 10    0.003879
 11    0.003018
 12    0.002362
 13    0.001868
 14    0.001483
 15    0.001185
 16    0.000951
 17    0.000766
 18    0.000619
 19    0.000502
 20    0.000408
 21    0.000332
 22    0.000271
 23    0.000222
 24    0.000182
 25    0.000150
 26    0.000123
 27    0.000102
 28    0.000084
iterations = 28.
```
