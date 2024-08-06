#pragma once
#include "utils.h"
#include "graph.h"
#include "samplegraph.h"
#include "sampling_utils.h"
#include <random>

/**
 * DeepWalk inputs:
 * graph G(V, E)
 * window size w (to be used in SkipGram)
 * walks per vertex \gamma
 * walk length t
*/

/**
 * @param g is a reference to the graph
 * @param transits is the current transit vertex
 * @param src_degree is the degree of the current transit vertex
 * @param step is the current step
 * @returns a vertex to add to s
*/
inline vidType sample_next(Graph &g, vidType transit, vidType src_degree, int step) {
    if (transit == (numeric_limits<uint32_t>::max)()) { return (numeric_limits<uint32_t>::max)(); }
    if (src_degree == 0) { return (numeric_limits<uint32_t>::max)(); }
    int idx = gen() % src_degree;
    return g.N(transit, idx);
}

/**
 * @param step
 * @param s is the sample
 * @param transitIdx is the index of transit of all transits to return
 * @returns the vertices added at previous step as transits
*/
vidType step_transits(int step, Sample * s, int transitIdx) {
    return s->prev_vertex(1, transitIdx);
}