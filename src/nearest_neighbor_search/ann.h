#pragma once
#include <vector>
#include "graph.h"

typedef std::vector<float> Embedding;
void ANN(Graph &g, std::vector<Embedding> points, Embedding query);

inline std::vector<Embedding> generate_embeddings(vidType num, int dim) {
  std::vector<Embedding> points(num);
  for (auto pt : points) {
    pt.resize(dim);
    for (int j = 0; j < dim; j++)
      pt[j] = rand() / (RAND_MAX + 1.0);
  }
  return points;
}

inline Embedding generate_query(int dim) {
  Embedding pt(dim);
  for (int j = 0; j < dim; j++)
    pt[j] = rand() / (RAND_MAX + 1.0);
  return pt;
}

