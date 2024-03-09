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
        e = floor(x);
        y = distribution(generator) * Q_v;
    }
    return v_edges[e];
}

/**
 * 
*/
inline vidType sample_next(Graph &g, vidType v, vidType t) {
    vector<vidType> t_edges;
    // only if previous transit exists do we find its related edges
    if (t != (numeric_limits<uint32_t>::max)()) {
        vidType *t_edge_ptrs = g.adj_ptr(t);
        vector<vidType> t_edges(t_edge_ptrs, t_edge_ptrs + sizeof t_edge_ptrs / sizeof t_edge_ptrs[0]);
    }
    vidType *v_edge_ptrs = g.adj_ptr(v);
    vector<vidType> v_edges(v_edge_ptrs, v_edge_ptrs + sizeof v_edge_ptrs / sizeof v_edge_ptrs[0]);
    float p = 2.0, q = 0.5;
    return rejection_smpl(v, v_edges, t, t_edges, p, q);
}

/**
 * Returns transit vertex for a sample at a given step and position
*/
inline vidType step_transits(int step, Sample* s, int transit_idx) {
    return s->prev_vertex(1, transit_idx);
}
