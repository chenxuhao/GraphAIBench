
# Graph Neural Networks

This is a parallel implementation of state-of-the-art Graph Neural Network (GNN) architectures, including Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), GraphSAGE, and Gated Graph Neural Networks (GGNNs) using OpenMP and CUDA.
Please see [1] for evaluation details.


## Table of Contents  

[Build](#build)  
[Datasets](#datasets)  
[Run](#run)  
[Supported GNN Model Architectures](#supported-gnn-model-architectures)      
[Performance Evaluation: Convergence Speed](#performance-evaluation-convergence-speed)   
[Publications](#publications)     
[Developers](#developers)     
[References](#references)      
 


<a name="build"></a>

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
<a name="datasets"></a>

## Datasets

Currently available datasets (please contact the author for access):

Single-class datasets:

-   tester
-   cora
-   citeseer
-   pubmed
-   reddit
-   flickr

Multi-class datasets:

-   ppi
-   yelp
-   amazon

The datasets are stored in binary format, as explained below:

`graph.meta.txt` text file specifying the meta information of the graph, including the number of vertices and edges; number of bytes for vertex IDs, edge IDs, vertex labels, and edge labels; maximum degree; feature vector length; distinct vertex label count; distinct edge label count; start index, end index and count of train/validation/test vertices.

`graph.vertex.bin` binary file containing the row pointers, with data type of edge IDs.

`graph.edge.bin` binary file containing the column indices, with data type of vertex IDs.

`graph.vlabel.bin` binary file containing the vertex labels (only needed for vertex labeled graphs)

`graph.elabel.bin` binary file containing the edge labels (only needed for edge labeled graphs)

`graph.feats.bin` binary file containing the vertex feature vectors (used for graph machine learning)

`train.masks.bin` binary file containing the masks for train vertex set

`val.masks.bin` binary file containing the masks for validation vertex set

`test.masks.bin` binary file containing the masks for test vertex set

<a name="run"></a>

## Run

To run the program after compilation, execute:

```
./gpu_train_gcn <dataset> <num_epochs> <num_threads> <type_loss> <size_hid> <num_layers> <size_subg> <size_frontier> <rate_learn>
```

where the first 4 arguments are mandatory. If the other arguments are not provided, the program will use the default value set in `./include/global.h`.

-   Our C++ training by default uses single precision floating point numbers.
-   You can set `<num_thread>` to be the number of physical cores in your system.
-   `<type_loss>` can be either `sigmoid` or `softmax`, and can affect accuracy significantly if not set correctly.

Examples:

CPU train using GCN model On Reddit dataset:

```
./cpu_train_gcn reddit 200 36 softmax
```

GPU trainning using GrapgSAGE model On PPI dataset:

```
./gpu_train_sage ppi 200 36 sigmoid
```



<a name="supported-gnn-model-architectures"></a>


## Supported GNN Model Architectures

-   [x] Graph Convolutional Networks (GCN)
-   [x] GraghSAGE (SAGE)
-   [x] Graph Attension Networks (GAT)
-   [x] Gated Graph Neural Networks (GGNN)

|      |             OpenMP |               CUDA |
| ---- | -----------------: | -----------------: |
| GCN  | :heavy_check_mark: | :heavy_check_mark: |
| GAT  | :heavy_check_mark: | :heavy_check_mark: |
| SAGE | :heavy_check_mark: | :heavy_check_mark: |
| GGNN |  | :heavy_check_mark: |


### Graph Convolutional Networks (GCNs)

GCNs, proposed by Kipf & Welling [2], are one of the most prominent graph neural network models. The fundamental concept behind GCNs is to learn a function of signals/features on a graph through the layer-wise propagation rule. Each node in the graph is represented by a feature vector, and these features are aggregated from neighboring nodes to generate a new feature representation. This process is repeated for several layers, facilitating information propagation throughout the graph.


### Graph Attention Networks (GATs)

GATs, introduced by Velickovic et al. [3], extend GCNs by incorporating an attention mechanism. This attention mechanism allows each node to assign different importances to its neighbors, thereby capturing the most relevant features in the aggregation process. The GAT model is especially advantageous in cases where the structure of the graph is inhomogeneous, and the relevance of neighboring nodes may significantly differ.


### GraphSAGE

GraphSAGE, presented by Hamilton et al. [4], introduces a novel approach that generalizes inductive learning process on large graphs. It generates low-dimensional vector representations for nodes, or embeddings, by aggregating features from a node's local neighborhood. The unique characteristic of GraphSAGE is its ability to generate embeddings for unseen data, hence enabling inductive learning on large and dynamic graphs.

### Gated Graph Neural Networks (GGNNs)

GGNNs, proposed by Li et al. [5], are a form of recurrent graph neural networks where node representations are updated based on their neighbors via gated recurrent units (GRUs). The recurrent nature of GGNNs allows for iterative refinement of node features and makes this model particularly suited for tasks where the information of the graph changes dynamically or where multiple steps of reasoning over the graph are required.

### Forward propagation

| Arch   | Scatter and ApplyEdge | Gather | ApplyVertex |
|--------|--------------|---------|-------------|
||$\psi( ... )$|$\oplus(...)$|$\phi(...)$|
| GCNs   |  $\frac{1}{\sqrt{d_id_j}}h_j$ |  $\sum_{j \in N(i)} \psi(h_j)$| $W \times (\oplus)$ |
| GATs   | ![equation](https://latex.codecogs.com/svg.image?%20%5Cfrac%7Bexp(%5Csigma(%5Cvec%7Ba%7D%5ET%20%5BW%20%5Cvec%7Bh%7D_i%20%5D%20%7C%7C%20W%5Cvec%7Bh%7Dj%5D))%7D%7B%5Csum%7Bk%20%5Cin%20N_i%7Dexp(%5Csigma(%5Cvec%7Ba%7D%5ET%20%5BW%5Cvec%7Bh%7D_i%20%7C%7C%20W%5Cvec%7Bh%7D_k%5D))%7D) | ![equation](https://latex.codecogs.com/svg.image?W%20%5Ctimes%20%5Csum_%7Bj%20%5Cin%20N(j)%7D%20%5Cpsi(h_i,h_j)%20) | $\sigma(\oplus h_i)$ |
|SAGE-GCN| $\frac{1}{d_i}h_j$ |$\sum_{j \in N^{*}(i)} \psi(h_j)$| $W \times (\oplus)$ |
| GG-NNs | $h_v^{(1)} = \[x_v^T,0\]^T$ | ![equation](https://latex.codecogs.com/svg.image?a_v%5E%7B(t)%7D%20=%20%5Csum_%7Bv%5E%5Cprime%20%5Cin%20IN(v)%7D%20f(h_%7Bv%5E%5Cprime%7D%5E%7B(t-1)%7D,l_%7B(v%5E%5Cprime,v)%7D)%20&plus;%20%5Csum_%7Bv%5E%5Cprime%20%5Cin%20OUT(v)%7D%20f(h_%7Bv%5E%5Cprime%7D%5E%7B(t-1)%7D,l_%7B(v,v%5E%5Cprime)%7D)%20$%20%5C%5C%5C%5C%20where%20%5C%5C%5C%5C%20$%20f(h_%7Bv%5E%5Cprime%7D%5E%7B(t-1)%7D,l_%7B(v%5E%5Cprime,v)%7D)%20=%20A%5E%7B(l_%7B(v,v%5E%5Cprime)%7D)%7D%20h_%7Bv%5E%5Cprime%7D%5E%7B(t-1)%7D%20&plus;%20b%5E%7B(l_%7B(v,v%5E%5Cprime)%7D)%7D) | $z_v^t = \sigma(W^z a_v^{(t)} + U^z h_v^{(t-1)} )$ <br> $r_v^t = \sigma(W^r a_v^{(t)} + U^r h_v^{(t-1)} )$ <br> $\tilde{h_v^{(t)}} = tanh(W a_v^{(t)} + U(r_v^t \odot h_v^{(t-1)}))$ <br> $h_v^{(t)} = (1 - z_v^t) \odot h_v^{(t-1)} + z_v^t \odot \tilde{h_v^{(t)}}$ |

<a name="performance-evaluation-convergence-speed"></a>

## Performance Evaluation: Convergence Speed

TBD

<a name="publications"></a>

## Publications

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
<a name="developers"></a>

## Developers

-   [Xuhao Chen](https://chenxuhao.github.io/), MIT
-   [Loc Hoang](https://www.cs.utexas.edu/~loc/), KatanaGraph
-   [Adrian Scridon](https://adrianscridon.com/)

<a name="references"></a>

## References

[1] Hoang, L., Chen, X., Lee, H., Dathathri, R., Gill, G., & Pingali, K. (2021). Efficient Distribution for Deep Learning on Large Graphs. GNNSys 2021. Retrieved from https://www.cs.utexas.edu/~loc/papers/deepgalois_gnnsys.pdf

[2] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. Proceedings of the International Conference on Learning Representations (ICLR). Retrieved from https://arxiv.org/abs/1609.02907

[3] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. Proceedings of the International Conference on Learning Representations (ICLR). Retrieved from https://arxiv.org/abs/1710.10903

[4] Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. Advances in Neural Information Processing Systems (NeurIPS). Retrieved from https://arxiv.org/abs/1706.02216

[5] Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2016). Gated graph sequence neural networks. Proceedings of the International Conference on Learning Representations (ICLR). Retrieved from https://arxiv.org/abs/1511.05493



## License

> Copyright (c) 2021, MIT
> All rights reserved.
