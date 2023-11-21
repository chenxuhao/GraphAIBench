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
inline vidType rejection_smpl(vidType v, vector<vidType> v_edges, vidType t, vector<vidType> t_edges, float p, float q) {
    set<vidType> t_dsts(t_edges.begin(), t_edges.end());
    vector<float> P_d;
    float Q_v = 0;
    for (auto v_dst: v_edges) {
        float weight;
        if (v_dst == t) weight = 1 / p;
        else if (t_dsts.find(v_dst) != t_dsts.end()) weight = 1.0;
        else weight = 1 / q;
        P_d.push_back(weight);
        if (weight > Q_v) Q_v = weight;
    }
    float x  = distribution(generator) * v_edges.size();
    int e = floor(x);
    float y = distribution(generator) * Q_v;
    while (y > P_d[e]) {
        x = distribution(generator) * v_edges.size();
        y = distribution(generator) * Q_v;
    }
    return v_edges[e];
}

/**
 * 
*/
inline vidType sample_next(Sample* s, vector<vidType> transits, vector<vidType> src_edges, int step, uint_fast32_t random_n) {
    if (transits[0] == (numeric_limits<uint32_t>::max)()) { return (numeric_limits<uint32_t>::max)(); }
    if (src_edges.size() == 0) { return (numeric_limits<uint32_t>::max)(); }
    vidType t = s->prev_vertex(2, 0);
    vector<vidType> t_edges = s->prev_edges(2, 0);
    float p = 2.0, q = 0.5;
    return rejection_smpl(transits[0], src_edges, t, t_edges, p, q);
}

/**
 * Number of steps in the random walk
*/
inline int steps() {
    return 1000;
}

/**
 * For given step, return number of samples to take. Step of -1 for original sapmle transits
*/
inline int sample_size(int step) {
    return 1;
}


inline int num_samples() {
    return 32;
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


inline vector<vector<uint_fast32_t>> generate_randoms() {
    vector<vector<uint_fast32_t>> random_nums;
    int num_steps = sample_size(-1);
    for (int step = 0; step < steps(); step++) {
        num_steps *= sample_size(step);
        vector<uint_fast32_t> per_step;
        for (int idx = 0; idx < num_steps * num_samples(); idx++) {
            per_step.push_back(gen());
        }
        random_nums.push_back(per_step);
    }
    return random_nums;
}

inline vector<vector<uint_fast32_t>> generate_init_randoms() {
    vector<vector<uint_fast32_t>> init_rands;
    for (int s = 0; s < num_samples(); s++) {
        vector<uint_fast32_t> per_sample;
        for (int n = 0; n < sample_size(-1); n++) {
            per_sample.push_back(gen());
        }
        init_rands.push_back(per_sample);
    }
    return init_rands;
}