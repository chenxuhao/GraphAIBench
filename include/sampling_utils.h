// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#pragma once
#include "utils.h"
#include "graph.h"
#include "samplegraph.h"
#include <random>

static std::mt19937 gen(time(nullptr));
static std::default_random_engine generator;
static std::uniform_real_distribution<float> distribution(0.0,1.0);

/**
 * Creates the initial seeds.
 *
 * @param seeds_size size of initial sample
 * @param graph_size size of original graph (has to be at least seeds_size)
 * @return vector containing node ids
*/
inline vector<vidType> get_initial_transits(vidType seeds_size, vidType graph_size) {
  // set<vidType> node_ids;
  // vector<vidType> n_ids;
  // while (node_ids.size() < seeds_size) {
  //   auto sample_id = gen() % graph_size;
  //   node_ids.insert(sample_id);
  // }
  // n_ids.insert(n_ids.end(), node_ids.begin(), node_ids.end());
  vector<vidType> n_ids;
  for (int i = 0; i < seeds_size; i++) {
    n_ids.push_back(gen() % graph_size);
  }
  return n_ids;
}


inline void allocate_transits(vector<vector<uint_fast32_t>> &rands, int t_size) {
  vector<uint_fast32_t> new_layer(t_size, 0);
  rands.push_back(new_layer);
}



/**
 *
*/
inline bool is_directed() {
    return false;
}