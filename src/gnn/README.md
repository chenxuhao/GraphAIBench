# GraphAIBench

GraphAIBench is a C++ implemented Benchmark Suite for Graph Neural Networks (GNN).

GraphAIBench is parallelized using OpenMP and CUDA, same as [DGL](https://github.com/dmlc/dgl/), but runs much faster than DGL. 
Please see [1] for evaluation details.

Therefore, compared to DGL and [PyG](https://github.com/pyg-team/pytorch_geometric), GraphAIBench is better suited for evaluating specialized hardware design or low-level library (e.g. SpMM) implementations for GNNs.

[1] Loc Hoang, Xuhao Chen, Hochan Lee, Roshan Dathathri, Gurbinder Gill, Keshav Pingali.
Efficient Distribution for Deep Learning on Large Graphs, GNNSys 2021

## Build

To compile, first set the environment variable, 

```
source env.sh
```

Then run from the current directory:

```
make
```

Now you are ready to perform parallel GCN training on CPU.

To compile for GPU, you need CUDA toolkit installed.

```
m clean; m -j USE_GPU=1 gpu_train_gcn
```

## Dataset

Please see an example dataset in inputs/tester.

Currently available datasets (please contact the author for access):

Single-class datasets:

* tester
* cora
* citeseer
* pubmed
* reddit
* flickr

Multi-class datasets:

* ppi
* yelp
* amazon

The datasets are stored in binary or text format, as explained below:

#### `.csgr`

This is the input graph (in CSR format).

#### `-feats.bin`

2D row-major array of type float32. It stores the normalized input feature of all vertices (training + validation + test).

#### `-labels.txt`

2D row-major array of type int. It stores the labels of all vertices (training + validation + test).

#### `_mask.txt`

1D array of type bool. It stores the mask for each vertex (train/val/test).

## Run

To run the program after compilation, execute:

```
./gpu_train_gcn <dataset> <num_epochs> <num_threads> <type_loss> <size_hid> <num_layers> <size_subg> <size_frontier> <rate_learn>
```

where the first 4 arguments are mandatory. If the other arguments are not provided, the program will use the default value set in `./include/global.h`. 

* Our C++ training by default uses single precision floating point numbers. 
* You can set `<num_thread>` to be the number of physical cores in your system. 
* `<type_loss>` can be either `sigmoid` or `softmax`, and can affect accuracy significantly if not set correctly. 

Examples:

CPU train using GCN model On Reddit dataset:

```
./cpu_train_gcn reddit 200 36 softmax
```

GPU trainning using GrapgSAGE model On PPI dataset:

```
./gpu_train_sage ppi 200 36 sigmoid
```

## Publications ##

```
@inproceedings{DeepGalois,
  title={Efficient Distribution for Deep Learning on Large Graphs},
  author={Hoang, Loc and Chen, Xuhao and Lee, Hochan and Dathathri, Roshan and Gill, Gurbinder and Pingali, Keshav},
  booktitle={Workshop on Graph Neural Networks and Systems},
  volume={1050},
  pages={1-9},
  year={2021}
}
```

## Developers ##

* `Xuhao Chen`, Research Scientist, MIT, cxh@mit.edu

## License ##

> Copyright (c) 2021, MIT
> All rights reserved.

