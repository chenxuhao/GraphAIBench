# GraphAIBench

GraphAIBench is a C++ implemented Benchmark Suite for [Graph AI](https://chenxuhao.github.io/graphAI.html). 
It includes the following benchmarks:

+ Graph Neural Networks (GNN): GCN, GraphSAGE, GAT.
+ Centrality: Betweenness Centrality (BC).
+ Community: Community detection using Louvain algorithm.
+ Components: Connected Components (CC), Srtongly Connected Components (SCC).
+ Corness: k-core decomposition.
+ Flitering: Minimum Spanning Tree (MST), Triangulated Maximally Filtered Graph (TMFG), Planar Maximally Filtered Graph (PMFG).
+ Linear Assignment: Hungarian algorithm.
+ Link Analysis: PageRank (PR).
+ Link Prediction: Node2vec.
+ Mining [1]: triangle counting, clique finding, motif counting, frequent subgraph mining (FSM).
+ Sampling: Random walk.
+ Structure: Graph coarsening.
+ Traversal: Breadth-First Search (BFS) and Single-Source Shortest Paths (SSSP).

GraphAIBench is parallelized using OpenMP and CUDA, same as [DGL](https://github.com/dmlc/dgl/), but runs much faster than DGL. 
Please see [2] for evaluation details.

Therefore, compared to DGL and [PyG](https://github.com/pyg-team/pytorch_geometric), GraphAIBench is better suited for evaluating specialized hardware design or low-level library (e.g. SpMM) implementations for GNNs.

Datasets for pattern mining (e.g. triangle counting) are available [here](https://www.dropbox.com/sh/i1jq1uwtkcd2qo0/AADJck_u3kx7FeSR5BvdrkqYa?dl=0).
Datasets for collaborative filtering (i.e. bipartite graphs) are available [here](https://www.dropbox.com/sh/ufb0cdnoe0ul8ir/AAAeFvtCcjilKU85svSYNscia?dl=0).
Datasets for vertex programs (e.g. BFS, SSSP, CC, BC, PageRank) are available [here (directed)](https://www.dropbox.com/sh/74lijht72duadp9/AAAoRUMKpT9a6LufTj0B8jfDa?dl=0) and [here (undirected)](https://www.dropbox.com/sh/cw98s5uoyvbokgz/AABx2TQHX-NJqQgihRDgS9l-a?dl=0).
Datasets for graph neural networks (GNNs), e.g. GCN, GraphSAGE, GAT, are available [here](https://www.dropbox.com/sh/cc59e7sw4gv5q0k/AAC9JybvjTknupwXOpUrx6Fza?dl=0).
Please contact [the author](http://people.csail.mit.edu/xchen/) for more datasets.

[1] Xuhao Chen, Arvind.
[Efficient and Scalable Graph Pattern Mining on GPUs](https://arxiv.org/pdf/2112.09761.pdf), OSDI 2022

[2] Loc Hoang, Xuhao Chen, Hochan Lee, Roshan Dathathri, Gurbinder Gill, Keshav Pingali,
[Efficient Distribution for Deep Learning on Large Graphs](https://people.csail.mit.edu/xchen/docs/gnnsys-2021.pdf),
Workshop on Graph Neural Networks and Systems (GNNSys), 2021

## Getting Started ##

The document is organized as follows:

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [Supported graph formats](#supported-graph-formats)
* [Code Documentation](#code-documentation)
* [Reporting bugs and contributing](#reporting-bugs-and-contributing)
* [Notes](#notes)
* [Publications](#publications)
* [Developers](#developers)
* [License](#license)

### Requirements ###

* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) 11.1.1 or greater.
* GCC 8.3.1 or greater.

### Quick Start ###

Edit `env.sh` to let the libraries pointing to the right paths in your system, and then:

    $ source env.sh

Then just make in the root directory:

    $ make

Or go to each sub-directory, e.g. src/triangle, and then make:

    $ cd src/triangle; make

Binaries will be in the `bin` directory. 
For example, `tc_omp_base` is the OpenMP version of triangle counting on CPU, `tc_gpu_base` is the single GPU version, and `tc_multigpu` is the multi-GPU version.

Find out commandline format by running executable without argument:

    $ cd ../../bin
    $ ./tc_omp_base

Datasets are available [here](https://www.dropbox.com/sh/i1jq1uwtkcd2qo0/AADJck_u3kx7FeSR5BvdrkqYa?dl=0).
Run triangle counting with an undirected toy graph on CPU:

    $ ./tc_omp_base ../inputs/citeseer/graph
    
You can find the expected outputs in the README of each benchmark [see here for triangle](https://github.com/chenxuhao/GraphAIBench/blob/master/src/triangle/README.md).

To control the number of threads, set the following environment variable:

    $ export OMP_NUM_THREADS=[ number of cores in system ]


### Supported graph formats ###

The graph loading infrastructure understands the following formats:

+ `graph.meta.txt` text file specifying the meta information of the graph, including the number of vertices and edges; number of bytes for vertex IDs, edge IDs, vertex labels, and edge labels; maximum degree; feature vector length; distinct vertex label count; distinct edge label count; start index, end index and count of train/validation/test vertices.

+ `graph.vertex.bin` binary file containing the row pointers, with data type of edge IDs.

+ `graph.edge.bin` binary file containing the column indices, with data type of vertex IDs.

+ `graph.vlabel.bin` binary file containing the vertex labels (only needed for vertex labeled graphs)

+ `graph.elabel.bin` binary file containing the edge labels (only needed for edge labeled graphs)

+ `graph.feats.bin` binary file containing the vertex feature vectors (used for graph machine learning)

+ `train.masks.bin` binary file containing the masks for train vertex set 

+ `val.masks.bin` binary file containing the masks for validation vertex set 

+ `test.masks.bin` binary file containing the masks for test vertex set 

An example graph is in inputs/citeseer

Other graph input formats to be supported:

* Market (.mtx), [The University of Florida Sparse Matrix Collection](http://www.cise.ufl.edu/research/sparse/matrices/)
* Metis (.graph), [10th DIMACS Implementation Challenge](http://www.cc.gatech.edu/dimacs10/)
* SNAP (.txt), [Stanford Network Analysis Project](http://snap.stanford.edu/)
* Dimacs9th (.gr), [9th DIMACS Implementation Challenge](http://www.dis.uniroma1.it/challenge9/)
* The Koblenz Network Collection (out.< name >), [The Koblenz Network Collection](http://konect.uni-koblenz.de/)
* Network Data Repository (.edges), [Network Data Repository](http://networkrepository.com/index.php)
* Real-World Input Graphs (Misc), [Real-World Input Graphs](http://gap.cs.berkeley.edu/datasets.html)

### Code Documentation ###

The code documentation is located in the `docs` directory (*doxygen* html format).

### Reporting bugs and contributing ###

If you find any bugs please report them by using the repository (github **issues** panel).
We are also ready to engage in improving and extending the framework if you request new features.

## Notes ##

Existing state-of-the-art frameworks:

Pangolin [1]: source code is in src/pangolin/

PBE [2,3]: https://github.com/guowentian/SubgraphMatchGPU

Peregrine [4]: https://github.com/pdclab/peregrine

Sandslash [5]: source code is in src/\*/cpu_kernels/\*_cmap.h

FlexMiner [6]: the CPU baseline code is in \*/cpu_kernels/\*_base.h

DistTC [7]: source code is in src/triangle/

DeepGalois [8]: https://github.com/chenxuhao/GraphAIBench

GraphPi [9]: https://github.com/thu-pacman/GraphPi

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali.
Pangolin: An Efficient and Flexible Graph Pattern Mining System on CPU and GPU. VLDB 2020

[2] Wentian Guo, Yuchen Li, Mo Sha, Bingsheng He, Xiaokui Xiao, Kian-Lee Tan.
GPU-Accelerated Subgraph Enumeration on Partitioned Graphs. SIGMOD 2020.

[3] Wentian Guo, Yuchen Li, Kian-Lee Tan. 
Exploiting Reuse for GPU Subgraph Enumeration. TKDE 2020.

[4] Kasra Jamshidi, Rakesh Mahadasa, Keval Vora.
Peregrine: A Pattern-Aware Graph Mining System. EuroSys 2020

[5] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Loc Hoang, Keshav Pingali.
Sandslash: A Two-Level Framework for Efficient Graph Pattern Mining, ICS 2021

[6] Xuhao Chen, Tianhao Huang, Shuotao Xu, Thomas Bourgeat, Chanwoo Chung, Arvind.
FlexMiner: A Pattern-Aware Accelerator for Graph Pattern Mining, ISCA 2021

[7] Loc Hoang, Vishwesh Jatala, Xuhao Chen, Udit Agarwal, Roshan Dathathri, Grubinder Gill, Keshav Pingali.
DistTC: High Performance Distributed Triangle Counting, HPEC 2019

[8] Loc Hoang, Xuhao Chen, Hochan Lee, Roshan Dathathri, Gurbinder Gill, Keshav Pingali.
Efficient Distribution for Deep Learning on Large Graphs, GNNSys 2021

[9] Tianhui Shi, Mingshu Zhai, Yi Xu, Jidong Zhai. 
GraphPi: high performance graph pattern matching through effective redundancy elimination. SC 2020

## Publications ##

Please cite the following paper if you use this code:

```
@article{Pangolin,
	title={Pangolin: An Efficient and Flexible Graph Mining System on CPU and GPU},
	author={Xuhao Chen and Roshan Dathathri and Gurbinder Gill and Keshav Pingali},
	year={2020},
	journal = {Proc. VLDB Endow.},
	issue_date = {August 2020},
	volume = {13},
	number = {8},
	month = aug,
	year = {2020},
	numpages = {12},
	publisher = {VLDB Endowment},
}
```

```
@INPROCEEDINGS{FlexMiner,
  author={Chen, Xuhao and Huang, Tianhao and Xu, Shuotao and Bourgeat, Thomas and Chung, Chanwoo and Arvind},
  booktitle={2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)}, 
  title={FlexMiner: A Pattern-Aware Accelerator for Graph Pattern Mining}, 
  year={2021},
  volume={},
  number={},
  pages={581-594},
  doi={10.1109/ISCA52012.2021.00052}
}
```

```
@inproceedings{DistTC,
  title={DistTC: High performance distributed triangle counting},
  author={Hoang, Loc and Jatala, Vishwesh and Chen, Xuhao and Agarwal, Udit and Dathathri, Roshan and Gill, Gurbinder and Pingali, Keshav},
  booktitle={2019 IEEE High Performance Extreme Computing Conference (HPEC)},
  pages={1--7},
  year={2019},
  organization={IEEE}
}
```

```
@inproceedings{Sandslash,
  title={Sandslash: a two-level framework for efficient graph pattern mining},
  author={Chen, Xuhao and Dathathri, Roshan and Gill, Gurbinder and Hoang, Loc and Pingali, Keshav},
  booktitle={Proceedings of the ACM International Conference on Supercomputing},
  pages={378--391},
  year={2021}
}
```

```
@inproceedings{hoang2019disttc,
  title={DistTC: High performance distributed triangle counting},
  author={Hoang, Loc and Jatala, Vishwesh and Chen, Xuhao and Agarwal, Udit and Dathathri, Roshan and Gill, Gurbinder and Pingali, Keshav},
  booktitle={2019 IEEE High Performance Extreme Computing Conference (HPEC)},
  pages={1--7},
  year={2019},
  organization={IEEE}
}
```

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

* [Xuhao Chen](http://people.csail.mit.edu/xchen/), Research Scientist, MIT, cxh@mit.edu
* [Tianhao Huang](https://nicsefc.ee.tsinghua.edu.cn/people/tianhao-h/), PhD student, MIT

## License ##

> Copyright (c) 2021, MIT
> All rights reserved.
