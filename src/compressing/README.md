# Sampling with Compressed Graphs
This part of the codebase is where I save and load compressed graphs using the streamvbyte method to use with sampling algorithms (so far just khop). For more details on the graph compression technique, refer to the codebase under src/structure.

To build the necessary files, from this directory (src/compressing) run 
```
source ../../env.sh
make clean
make
```

The command to sample using any of the following methods below follows the same structure of the executable followed by the 4 arguments:
```
../../bin/<OBJECT_FILE> <PATH_TO_GRAPH>/<INPUT_GRAPH_PREFIX> <PATH_TO_GRAPH>/<OUTPUT_GRAPH_PREFIX> <NUM_OF_BATCHES> <NUM_OF_THREADS>
```
Note that all our graphs are stored in csr format. Each graph requires 3 files: (1) <GRAPH_PREFIX>.meta.txt (2) <GRAPH_PREFIX>.vertex.bin (3) <GRAPH_PREFIX>.edge.bin

The number of batches is how many initial transits our first khop layer starts with. Number of threads is for parallel and gpu sampling only (can omit for serial or just leave it). Other parameters of khop sampling such as expansion size per transit or number of layers can be directly edited in include/khop.h or include/khop_gpu.cuh (but make sure to change both to match for gpu code to work correctly!!).

An example command is provided for each of the following sampling methods below using the uk2007 graph with a batch size of 40000 and 64 threads (per block for gpu).

## Sampling on CPU
### khop in serial
To run khop sampling in serial (one thread), run
```
../../bin/cpu_serial_khop /projects/bbof/chen27/automine-inputs/uk2007/graph /projects/bbof/chen27/automine-inputs/uk2007/vbyte 40000
```

### parallel khop
To run khop sampling in parallel using OpenMP, run
```
../../bin/cpu_omp_khop /projects/bbof/chen27/automine-inputs/uk2007/graph /projects/bbof/chen27/automine-inputs/uk2007/vbyte 40000 64
```

## Sampling on GPU
Before sampling on gpu, make sure gpu are available. Check by running `nvidia-smi` and a table of gpu information should output if present. If not, run a slurm job to request. On delta, run the example under "Start a job" section located in this doc: https://docs.google.com/document/d/1MGrXJXV2Q9bgfStQuEVWpDxoJXdNE1hDCgldRVPdFaI/edit#heading=h.rp7lyrv7xmlt. The account name may need to be changed depending on permissions.

### allocating compressed graph onto gpu
To run khop sampling on the gpu with the entire compressed graph stored on gpu memory, run
```
../../bin/on_gpu_khop /projects/bbof/chen27/automine-inputs/uk2007/graph /projects/bbof/chen27/automine-inputs/uk2007/vbyte 40000 64
```
This gpu version has the benefits of fast decompression on gpu with low communication overhead, but has the drawback of limited memory for the graph.

### using unified memory (TBD)
this version will store compressed graph on unified memory and communicate compressed transit neighborhood when needed, will update when finished

## Creating a compressed graph
In any of the main .cu or .cuh files, there should be a commented out line
```
// save_compressed_graph(in_prefix, out_prefix);
```
If you do not already have a compressed version of the graph you are passing in, then uncomment this line and the sampling process will include creating and saving a compressed version of your input to the output location provided by the described first two arguments in the command line.