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


inline vidType sample_next(Sample &s, vidType transit, vidType src_degree, int step) {
    if (transit == (numeric_limits<uint32_t>::max)()) { return (numeric_limits<uint32_t>::max)(); }
    if (src_degree == 0) { return (numeric_limits<uint32_t>::max)(); }
    int idx = gen() % src_degree;
    return s.get_graph()->N(transit, idx);
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

inline vidType sample_next_fixed(Sample* s, vector<vidType> transits, vector<vidType> src_edges, int step, uint_fast32_t rand_idx) {
    if (transits[0] == (numeric_limits<uint32_t>::max)()) { return (numeric_limits<uint32_t>::max)(); }
    if (src_edges.size() == 0) { return (numeric_limits<uint32_t>::max)(); }
    int idx = rand_idx % src_edges.size();
    return src_edges[idx];
}

/**
 * Number of steps in the random walk
*/
inline int steps() {
    return 2;
}

/**
 * For given step, return number of samples to take. Step of -1 for original sapmle transits
*/
inline int sample_size(int step) {
    if (step == -1) return 1;
    if (step == 0) return 25;
    return 10;
    // if (step == -1) return 2;
    // return 2;
}


inline int num_samples() {
    return 40000;
    // return 2;
}


/**
 * For given step, should sample only contain unique vertices
*/
inline bool unique(int step) {
    return false;
}

/**
 * Type of transit sampling
*/
inline SamplingType sampling_type() {
    return Individual;
}

/**
 * Returns transit vertex for a sample at a given step and position
*/
inline vidType step_transits(int step, Sample* s, int transit_idx) {
    return s->prev_vertex(1, transit_idx);
}

