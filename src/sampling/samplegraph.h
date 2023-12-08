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
    std::vector<vidType> all_transits;
    int filled_layers;
    Graph* g;

public:
    Sample(int num, Graph* graph) {
        // transits_order.push_back(inits);
        // g = graph;
        // filled_layers = 1;
        allocate_transits(num);
        g = graph;
    };

    void allocate_layer(int num_transits) {
        std::vector<vidType> layer(num_transits, 0);
        transits_order.push_back(layer);
    }

    void allocate_transits(int num) {
        std::vector<vidType> empty_transits(num, 0);
        all_transits = empty_transits;
    }

    void add_inits(vector<vidType> inits) {
        for (int i = 0; i < inits.size(); i++) { all_transits[i] = inits[i]; }
    }

    void increment_filled_layer() {
        filled_layers++;
    }

    int get_layer_count() {
        return filled_layers;
    }

    void write_transit(int pos, vidType t_value) {
        // transits_order[filled_layers][pos] = t_value;
        all_transits[pos] = t_value;
    }

    vidType prev_vertex(int i, int pos) {
        // int step = filled_layers - i;
        // if (step < 0) return prev_vertex(1, pos);
        // vidType transit = transits_order[step][pos];
        // return transit;
        return all_transits[pos];
    }


    vidType prev_vertex_degree(int i, vidType v) {
        // int step = filled_layers - i;
        // if (step < 0) return prev_vertex_degree(1, pos);
        // int transit = transits_order[step][pos];
        // return g->out_degree(transit);
        return g->out_degree(v);
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

    std::vector<vidType> get_all_transits() {
        return all_transits;
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
