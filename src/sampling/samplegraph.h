#pragma once
#include "VertexSet.h"

using namespace std;

enum SamplingType {
    Individual,
    Collective,
};

class Sample {
protected:
    std::unordered_map<vidType, std::set<vidType>> edges; // parent mapping only for importance sampling
    std::vector<std::vector<vidType>> transits_order;
    int filled_layers;
    Graph* g;

public:
    Sample(std::vector<vidType> inits, Graph* graph) {
        transits_order.push_back(inits);
        g = graph;
        filled_layers = 1;
    };

    void allocate_layer(int num_transits) {
        std::vector<vidType> layer(num_transits, 0);
        transits_order.push_back(layer);
    }

    void increment_filled_layer() {
        filled_layers++;
    }

    int get_layer_count() {
        return filled_layers;
    }

    void write_transit(int pos, vidType t_value) {
        transits_order[filled_layers][pos] = t_value;
    }

    vidType prev_vertex(int i, int pos) {
        int step = filled_layers - i;
        if (step < 0) return prev_vertex(1, pos);
        vidType transit = transits_order[step][pos];
        return transit;
    }

    std::vector<vidType> prev_edges(int i, int pos) {
        int step = filled_layers - i;
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

    // void add_transits(std::vector<vidType> new_ts) { transits_order.push_back(new_ts); }

    // root functions only for multidimensional random walks
    std::vector<vidType> get_transits() {
        int step = filled_layers - 1;
        if (step < 0) return transits_order[0];
        return transits_order[step];
    }

    Graph* get_graph() { return g; }

    void copy_transits() {
        int idx = 0;
        std::vector<vidType> new_roots;
        for (auto r: get_transits()) {
            write_transit(idx++, r);
        }
    }

    // only for importance sampling
    void add_edge(vidType parent, vidType v) {
        edges[parent].insert(v);
    }

    std::unordered_map<vidType, std::set<vidType>> get_edges() { return edges; }
    std::vector<std::vector<vidType>> get_transits_order() { return transits_order; }
    // std::vector<vidType> get_transits() {return transits_order[transits_order.size() - 1];}
    // std::vector<vidType> get_transit_edges() {return transits_order[transits_order.size() - 2];}
    // int get_steps_taken() {return steps_taken;}
};
