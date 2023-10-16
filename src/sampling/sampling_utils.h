// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "utils.h"
#include "graph.h"
#include "samplegraph.h"
#include <random>
std::mt19937 gen(time(nullptr));
std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0.0,1.0);

/**
 * Creates the initial seeds.
 *
 * @param seeds_size size of initial sample
 * @param graph_size size of original graph (has to be at least seeds_size)
 * @return vector containing node ids
*/
inline vector<vidType> get_initial_transits(uint64_t seeds_size, uint64_t graph_size) {
  set<uint32_t> node_ids;
  vector<uint32_t> n_ids;
  while (node_ids.size() < seeds_size) {
    auto sample_id = gen() % graph_size;
    node_ids.insert(sample_id);
  }
  n_ids.insert(n_ids.end(), node_ids.begin(), node_ids.end());
  return n_ids;
}


/**
 *
*/
inline bool is_directed() {
    return false;
}