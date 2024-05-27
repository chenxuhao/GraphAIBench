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
../../bin/gpu_uncomp ~/data-xhchen/mcai1/inputs/uk2007/order `-u`
```

### using normal compressed graph
```
# in-memory
../../bin/gpu_vbyte_warp ~/data-xhchen/mcai1/inputs/uk2007/order-vbyte -s 0

# on uva
../../bin/gpu_vbyte_warp ~/data-xhchen/mcai1/inputs/uk2007/order-vbyte -s 1
```

### using prefix sums compressed subgraphs
Here, we have a `config.txt` that has four lines, one line for each subgraph that should have *true* if move to uva, *false* if leave in-memory. The lines are in order of top degree, high degree, medium degree, and low degree subgraphs respectively. Note that there are four partitions in this code but only two subgraphs (just low $G_{l}$ and high $G_{h}$) in our paper. This is because of a prior implementation with more subgraphs, but now subgraphs medium, high, and top are all encoded and decoded the same to make up $G_{h}$ in the paper.

The below commands are for loading in pre-saved subgraphs. For details on how to save subgraphs, go to section on saving subgraphs. Flags `-l`, `-h`, `-u` must match degree thresholds from when graphs were originally made, and can be found in the saved file names.
```
# without warm-up kernel to cache G_l
../../bin/gpu_vbyte_warp ~/data-xhchen/mcai1/inputs/uk2007/type4/ -r -s 2 -l 32 -h 160 -u 256

# with warm-up kernel to cache G_l
../../bin/gpu_vbyte_warp ~/data-xhchen/mcai1/inputs/uk2007/type4/ -r -k -s 2 -l 32 -h 160 -u 256
```

The `config.txt` should look like the following for different versions, with **no comments or extra lines** other than the four booleans.
```
# in-memory
false
false
false
false

# low parition (G_l) on uva
false
false
false
true
```

## Creating a compressed graph
In `gpu_vbyte_warp.cu`, there is a `-c` flag you can use during execution to create a compressed graph if it doesn't exist already. 
The program will then create and save a compressed version of your input to the output location provided by the described first two arguments in the command line.

## Saving subgraphs
`save_subgraphs.cc` contains the code to generate the different type of subgraphs. There's a flag `-a` that encodes the top degree subgraph the same as the medium and high (using prefix sums compression). With this flag, we basically have the low, normal compressed partition $G_{l}$ and high, prefix sums partition $G_{h}$ (just split into med, high, top but treated the same) from the paper.

When creating subgraphs, we also want to specify the degree thresholds that divides nodes into one of each partition. Only `-l`, the low degree threshold flag, really matters to create $G_{l}$ and $G_{h}$. The other flags `-h` and `-u` can be anything as long as `-h` is greater than `-l` and `-u` is greater than `-h` because again the med, high, and top partitions in this codebase are all treated the same. From the paper, we chose the lowest multiple of 32 for `-l` that would allow $G_{h}$ to fit into GPU memory. Finding this number is not yet automated but by trial and error. 

```
# smaller graph that completely fits in GPU memory, choose smallest -l=32
../../bin/save_subgraphs ~/data-xhchen/mcai1/inputs/uk2007/ -a -l 32 -h 160 -u 256

# larger graph where the `-l` flag matters
../../bin/save_subgraphs ~/data-xhchen/mcai1/inputs/clueweb12/ -a -l 128 -h 160 -u 256
```

## Datasets
There are already saved original graphs (graph.\*), relabeled graphs (order.\*), streamvbyte compressed graphs (order-vbyte.\*), and subgraphs (u\.*, l.\*, h.\*, m.\*) in the corresponding graph directories under ~/data-xhchen/mcai1/inputs/. While our hybrid sampling method only uses 2 subgraphs in the paper, it is split into 4 subgraphs (low, medium, high, and top) in this codebase due to convenience from prior experiments. The medium, high, and top subgraphs are all just treated the same now (the high subgraph in the paper)

 For larger graphs that require big memory, run the bigmem.sh executable in this directory to request allocation for cpu operations.