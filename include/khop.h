// Copyright 2024 MIT
// Authors: Luc Gaitskell <lucg@mit.edu>
#pragma once
#include "utils.h"
#include "graph.h"
#include <random>
std::mt19937 gen(time(nullptr));
// std::default_random_engine generator;
// std::uniform_real_distribution<float> distribution(0.0,1.0);

inline vidType sample_next_vbyte(Graph &g, vidType transit)
{
    auto adj_transit = g.N_vbyte(transit, "streamvbyte");
    vidType src_degree = adj_transit.size();
    if (src_degree == 0)
    {
        return (numeric_limits<uint32_t>::max)();
    }
    int idx = gen() % src_degree;
    return adj_transit.data()[idx];
}

inline vidType sample_next(Graph &g, vidType transit, vidType src_degree, int step)
{
    if (transit == (numeric_limits<uint32_t>::max)())
    {
        return (numeric_limits<uint32_t>::max)();
    }
    if (src_degree == 0)
    {
        return (numeric_limits<uint32_t>::max)();
    }
    int idx = gen() % src_degree;
    // int idx = 1;
    return g.N(transit, idx);
}
