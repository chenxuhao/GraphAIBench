# Sampling with Compressed Graphs
This part of the codebase is where I save and load compressed graphs using the streamvbyte method to use with sampling algorithms (so far just khop). For more details on the graph compression technique, refer to the codebase under src/structure.

To build the necessary files, from this directory (src/compressing) run 
```
source ../../env.sh
make clean
make
```
Depending on the server, you may need to run another command for nvcc
```
# rockfish
module load cuda/11.8.0

# anvil
source /etc/profile.d/modules.sh
module load modtree/gpu
module load gcc/11.2.0
```

Note that all our graphs are stored using the following 3 files (even compressed subgraphs): (1) <GRAPH_PREFIX>.meta.txt (2) <GRAPH_PREFIX>.vertex.bin (3) <GRAPH_PREFIX>.edge.bin. We also require that all input graphs be relabeled first such that they are in order of decreasing degrees. This is better to be done on anvil as a background sbatch job since it can take hours and requires big memory for larger input graphs.

The number of batches is how many initial transits our first khop layer starts with. Other parameters of khop sampling such as expansion size per transit or number of steps can be directly edited in include/sampling_utils.h.

An example command is provided for each of the following sampling methods below using the uk2007 graph with a batch size of 40000 and 256 threads (per block for gpu). Examples are for **ROCKFISH ONLY since we need the A100 GPU with 80GB memory.

## Sampling on CPU
### parallel khop
To run khop sampling in parallel using OpenMP (with batch size of 40000 and 32 threads) for a compressed graph, run
```
../../bin/cpu_omp_khop ~/data-xhchen/mcai1/inputs/uk2007/order-vbyte 40000 32
```
To do so for an *uncompressed* graph, go to the sampling directory and run
```
cd ../sampling
make khop_omp
../../bin/khop_omp ~/data-xhchen/mcai1/inputs/uk2007/order 40000 32
```

## Sampling on GPU
Before sampling on gpu, make sure gpu are available. Check by running `nvidia-smi` and a table of gpu information should output if present. If not, run a slurm job to request. On rockfish, use the gpu.sh executable. 


### using uncompressed graph
```
# in-memory
../../bin/gpu_uncomp ~/data-xhchen/mcai1/inputs/uk2007/order

# on uva
../../bin/gpu_uncomp ~/data-xhchen/mcai1/inputs/uk2007/order -u
```

### using normal compressed graph
```
# in-memory
../../bin/gpu_vbyte_warp ~/data-xhchen/mcai1/inputs/uk2007/order-vbyte -s 0

# on uva
../../bin/gpu_vbyte_warp ~/data-xhchen/mcai1/inputs/uk2007/order-vbyte -s 1
```

### using prefix sums compressed subgraphs
Here, we have a config.txt that has four lines, one line for each subgraph that should have *true* if move to uva, *false* if leave in-memory. The lines are in order of top degree, high degree, medium degree, and low degree subgraphs respectively. Note that there are four subgraphs in code but only two subgraphs (just low and high) in our paper. This is because of a prior implementation with more subgraphs, but now subgraphs medium, high, and top are all encoded and decoded the same to make up the high subgraph in the paper.

The below commands are for loading in pre-saved subgraphs. For details on how to save subgraphs, go to section on saving subgraphs. Flags -l, -h, -u must match degree thresholds from when graphs were originally made, can be found in the saved file names.
```
# in-memory
../../bin/gpu_vbyte_warp ~/data-xhchen/mcai1/inputs/uk2007/type4/ -r -s 2 -l 32 -h 160 -u 256
```

## Creating a compressed graph
In gpu_vbyte_warp.cu, there is a -c flag you can use during execution to create a compressed graph if it doesn't exist already. 
The program will then create and save a compressed version of your input to the output location provided by the described first two arguments in the command line.

## Datasets
There are already saved original graphs (graph.\*), relabeled graphs (order.\*), streamvbyte compressed graphs (order-vbyte.\*), and subgraphs (u\.*, l.\*, h.\*, m.\*) in the corresponding graph directories under ~/data-xhchen/mcai1/inputs/. While our hybrid sampling method only uses 2 subgraphs in the paper, it is split into 4 subgraphs (low, medium, high, and top) in this codebase due to convenience from prior experiments. The medium, high, and top subgraphs are all just treated the same now (the high subgraph in the paper)

 For larger graphs that require big memory, run the bigmem.sh executable in this directory to request allocation for cpu operations.