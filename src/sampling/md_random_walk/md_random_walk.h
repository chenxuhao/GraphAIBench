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
    int e_idx = gen() % src_edges.size();
    vidType v = src_edges[e_idx];
    s->copy_transits();
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
    if (step == -1) return 4;
    return 1;
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
    return Individual;
}

/**
 * Returns transit vertex for a sample at a given step and position
*/
inline vidType step_transits(int step, Sample* s, int transit_idx) {
    return s->get_transits()[transit_idx];
}