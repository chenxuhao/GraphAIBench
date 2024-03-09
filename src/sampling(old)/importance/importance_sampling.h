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


/**
 *
*/
inline vidType sample_next(Sample* s, vector<vidType> transits, vector<vidType> src_edges, int step) {
    vidType v = gen() % s->get_graph()->num_vertices();
    // bool connected = false;
    // for (auto trn: transits) {
    //     if (s->get_graph()->is_connected(v, trn)) {
    //         s->add_edge(trn, v);
    //         // connected = true;
    //         if (!is_directed()) s->add_edge(v, trn);
    //     }
    // }
    // if (connected) return v;
    return v;
}

inline tuple<vidType, uint_fast32_t> sample_next_store(Sample* s, vector<vidType> transits, vector<vidType> src_edges, int step) {
    uint_fast32_t rand_idx = gen();
    vidType v = rand_idx % s->get_graph()->num_vertices();
    return {v, rand_idx};
}

inline vidType sample_next_fixed(Sample* s, vector<vidType> transits, vector<vidType> src_edges, int step, uint_fast32_t rand_idx) {
    vidType v = rand_idx % s->get_graph()->num_vertices();
    return v;
}

/**
 * Number of steps in the random walk
*/
inline int steps() {
    return 3;
}

/**
 * For given step, return number of samples to take. Step of -1 for original sapmle transits
*/
inline int sample_size(int step) {
    return 2;
}


inline int num_samples() {
    return 2;
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
    return Collective;
}

/**
 * Returns transit vertex for a sample at a given step and position
*/
inline vidType step_transits(int step, Sample* s, int transit_idx) {
    return s->prev_vertex(1, transit_idx);
}