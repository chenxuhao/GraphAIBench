// Copyright 2024 MIT
// Authors: Luc Gaitskell <lucg@mit.edu>
#pragma once
#include "utils.h"
#include "graph.h"
#include <random>
std::mt19937 gen(time(nullptr));
// std::default_random_engine generator;
// std::uniform_real_distribution<float> distribution(0.0,1.0);

/**
 * Content below from Miranda Cai
 */

/**
 * Creates the initial seeds.
 *
 * @param seeds_size size of initial sample
 * @param graph_size size of original graph (has to be at least seeds_size)
 * @return vector containing node ids
 */
inline vector<vidType> get_initial_transits(vidType seeds_size, vidType graph_size)
{
    // set<vidType> node_ids;
    // vector<vidType> n_ids;
    // while (node_ids.size() < seeds_size) {
    //   auto sample_id = gen() % graph_size;
    //   node_ids.insert(sample_id);
    // }
    // n_ids.insert(n_ids.end(), node_ids.begin(), node_ids.end());
    vector<vidType> n_ids;
    for (int i = 0; i < seeds_size; i++)
    {
        n_ids.push_back(gen() % graph_size);
    }
    return n_ids;
}

/**
 * For given step, return number of samples to take. Step of -1 for original sample transits
 */
inline int sample_size(int step)
{
    if (step == -1)
        return 1;
    if (step == 0)
        return 25;
    return 10;
    // if (step == -1) return 2;
    // return 2;
}

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
