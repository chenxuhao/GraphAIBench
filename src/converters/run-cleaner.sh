#!/bin/bash

date
set -eu
export OMP_NUM_THREADS=64

../../bin/cleaner ~/scratch/undirected-graphs/wdc14/graph ~/scratch/undirected-graphs/wdc14-clean/graph
