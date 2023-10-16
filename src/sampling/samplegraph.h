#pragma once
#include "VertexSet.h"

using namespace std;

enum SamplingType {
    Individual,
    Collective,
};

class Sample {
protected:
    // std::vector<vidType> vertices;
    // std::vector<eidType> edges;
    // std::vector<std::unordered_map<vidType, vidType>> transits; parent mapping only needed for collective
    std::vector<std::vector<vidType>> transits_order;
    // int steps_taken;
    Graph* g;

public:
    Sample(std::vector<vidType> inits, Graph* graph) {
        transits_order.push_back(inits);
        // std::unordered_map<vidType, vidType> inits_map;
        g = graph;
        // steps_taken = 0;
        // node should never be a parent of itself, so this is default check
        // for (auto t: inits) inits_map[t] = t;
        // transits.push_back(inits_map);
    };

    vidType prev_vertex(int i, int pos) {
        int step = transits_order.size() - i;
        if (step < 0) return prev_vertex(1, pos);
        int transit = transits_order[step][pos];
        return transit;
    }

    std::vector<vidType> prev_edges(int i, int pos) {
        int step = transits_order.size() - i;
        if (step < 0) return prev_edges(1, pos);
        int transit = transits_order[step][pos];
        eidType edge_begin = g->edge_begin(transit);
        eidType edge_end = g->edge_end(transit);
        std::vector<vidType> edges;
        for (int i = edge_begin; i < edge_end; i++) {
            edges.push_back(g->colidx()[i]);
        }
        return edges;
    }

    void add_transits(std::vector<vidType> new_ts) { transits_order.push_back(new_ts); }

    // std::vector<vidType> get_transits() {return transits_order[transits_order.size() - 1];}
    // std::vector<vidType> get_transit_edges() {return transits_order[transits_order.size() - 2];}
    // int get_steps_taken() {return steps_taken;}