There are different categories of graph algorithms: 

(1) Graph analytics, e.g., PageRank (link analysis), SSSP, BFS, betweenness centrality, connected components. 
They are know as [vertex programs](https://shawnliu.me/post/vertex-centric-graph-processing-the-what-and-why/);

(2) Graph pattern mining (GPM), e.g., triangle counting, k-clique listing, motif counting, graph querying, frequent subgraph mining; They look for small patterns (e.g. traingle) in the graph, by using mostly set intersection and set difference operations. GPM algorithms often access k-hop neighbors, where k=n-1, and n is the number of vertices in the pattern graph.

(3) Graph machine learning (GML), e.g., graph embedding, graph neural networks, collaborative filtering, link prediction; In these algorithms, each vertex has a feature vector, instead of a scalar value, which could be expensive for data communication. 

(4) Graph sampling, e.g., random walk, neighbor sampling, subgraph sampling, layer-wise sampling, etc; They randomly sample the graph by rolling the dice for the next step when traversing the graph.

(5) Graph manipulation: graph construction/generation, graph partitioning, graph coarsening. They are wildly used tools and preprocesing step for other graph algorithms.

(6) Graph clustering: the well-known Louvain algorithm and its variances. The major operation is swapping nodes between clusters.

(7) Graph filtering: Triangulated Maximally Filtered Graph (TMFG), Minimum Spanning Tree (MST), Planar Maximally Filtered Graph (PMFG). They try to extract the backbone or core structure of the graph and ignore the rest.

(8) Streaming/dynamic graphs: vertex/edge insertion/deletion, B-tree, C-tree, Packed Memory Array (PMA). Graphs are evolving over time.
