// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#pragma once
#include "utils.h"
#include "graph.h"
#include "samplegraph.h"
#include "sampling_utils.h"
#include <random>
// std::mt19937 gen(time(nullptr));
// std::default_random_engine generator;
// std::uniform_real_distribution<float> distribution(0.0,1.0);

inline vidType sample_next_vbyte(Graph &g, vidType transit) {
    auto adj_transit = g.N_vbyte(transit, "streamvbyte");
    vidType src_degree = adj_transit.size();
    if (src_degree == 0) { return (numeric_limits<uint32_t>::max)(); }
    int idx = gen() % src_degree;
    return adj_transit.data()[idx];
}


inline vidType sample_next(Graph &g, vidType transit, vidType src_degree, int step) {
    if (transit == (numeric_limits<uint32_t>::max)()) { return (numeric_limits<uint32_t>::max)(); }
    if (src_degree == 0) { return (numeric_limits<uint32_t>::max)(); }
    int idx = gen() % src_degree;
    // int idx = 1;
    return g.N(transit, idx);
}

/**
 * 
*/
inline tuple<vidType, uint_fast32_t> sample_next_store(Sample* s, vector<vidType> transits, vector<vidType> src_edges, int step) {
    if (transits[0] == (numeric_limits<uint32_t>::max)()) { return {(numeric_limits<uint32_t>::max)(), 0}; }
    if (src_edges.size() == 0) { return {(numeric_limits<uint32_t>::max)(), 0}; }
    uint_fast32_t rand_idx = gen();
    int idx = rand_idx % src_edges.size();
    return {src_edges[idx], rand_idx};
}

inline vidType sample_next_fixed(Sample &s, vidType transit, vidType src_degree, int step, uint_fast32_t rand_idx) {
    if (transit == (numeric_limits<uint32_t>::max)()) { return (numeric_limits<uint32_t>::max)(); }
    if (src_degree == 0) { return (numeric_limits<uint32_t>::max)(); }
    int idx = rand_idx % src_degree;
    // int idx = 1;
    return s.get_graph()->N(transit, idx);
}

/**
 * Returns transit vertex for a sample at a given step and position
*/
inline vidType step_transits(int step, Sample* s, int transit_idx) {
    return s->prev_vertex(1, transit_idx);
}

