#pragma once
#include "lgraph.h"
#define ETA 1.5          // length factor of DB in sampling
#define SAMPLE_CLIP 3000 // clip degree in sampling

typedef std::set<index_t> VertexSet;
typedef std::vector<index_t> VertexList;

class Sampler {
protected:
  index_t m;          // number of vertice in the frontier
  size_t count_;
  int avg_deg;        // averaged degree of masked graph
  int subg_deg;       // average degree cut off to a clip
  Graph* full_graph;  // the original full graph
  Graph* masked_graph; // sampling set masked original graph; typically to the training set
  std::vector<index_t> trainingNodes; //! List of training nodes; 

  //! Reindex a graph to only contain those in the vertex set
  void reindexSubgraph(VertexSet& keptVertices, Graph& g, Graph& reindexed);
  //! Given a graph, return a graph with edges to unmasked vertices removed in mg
  template <typename GraphTy, typename SubgraphTy = Graph>
  void getMaskedGraph(index_t n, mask_t* masks, GraphTy* g, SubgraphTy* sub);
  //! determine degree of each vertex in a masked graph (given by masks and g)
  template <typename GraphTy = Graph>
  void getMaskedDegrees(size_t n, mask_t* masks, GraphTy* g, std::vector<uint32_t>& degrees);
  void createMasks(size_t n, VertexSet vertices, mask_t* masks) {
    std::fill(masks, masks + n, 0);
    for (auto v : vertices) masks[v] = 1;
  }
  //! helper function to get degree of some vertex given some graph
  inline unsigned getDegree(Graph* g, index_t v) {
    return g->edge_end_host(v) - g->edge_begin_host(v);
  }
  inline VertexList reindexVertices(size_t n, VertexSet vertex_set) {
    VertexList new_ids(n, 0);
    int vid = 0;
    for (auto v : vertex_set) {
      new_ids[v] = vid++; // reindex
    }
    return new_ids;
  }

public:
  Sampler(Graph* g, Graph* tg, mask_t* masks, size_t count) : 
    m(DEFAULT_SIZE_FRONTIER), count_(count), full_graph(g), masked_graph(tg) {
      // save ids (of original graph) of training nodes to vector
      for (size_t i = 0; i < full_graph->size(); i++)
        if (masks[i] == 1) trainingNodes.push_back(i);
      avg_deg = masked_graph->sizeEdges() / masked_graph->size();
      subg_deg = (avg_deg > SAMPLE_CLIP) ? SAMPLE_CLIP : avg_deg;
  }
  ~Sampler() {}
  void generateSubgraph(VertexSet& vertex_set, mask_t* masks, Graph* sg);
  size_t selectVertices(index_t nv, index_t n, Graph* g, VertexList vertices, VertexSet& vertex_set);
  size_t select_vertices(index_t n, VertexSet& vertex_set, unsigned seed);
};

