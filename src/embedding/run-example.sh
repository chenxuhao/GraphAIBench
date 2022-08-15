#!/bin/bash
../../bin/cf_omp_base ../../inputs/amazon-ratings/graph
#../../bin/cf_omp_base ../../inputs/netflix_mm.mtx 0.05 0.003

./bin/cf_omp_base ~/datasets/bipartite-graphs/netflix_mm/graph 0.001 0.00003
