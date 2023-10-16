#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "importance_sampling.h"
#include "samplegraph.h"
#include "sampling_utils.h"
using namespace std;

inline vidType sample_next(Sample* s, vector<vidType> transits, vector<vidType> src_edges, int step);
inline bool is_directed();
inline int steps();
inline int sample_size(int step);
inline int num_samples();
// inline bool unique(int step); assuming always false for now
inline SamplingType sampling_type();
inline vidType step_transits(int step, Sample* s, int transit_idx);
inline vector<vidType> get_initial_transits(uint64_t seeds_size, uint64_t graph_size);

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <graph>"
              << "[num_gpu(1)] [chunk_size(1024)]\n";
    cout << "Example: " << argv[0] << " ../inputs/cora/graph\n";
    exit(1);
  }

  // create graph and retrieve node/edge data
  Graph g(argv[1], 0, 0, 0, 0, 0);
  eidType* rptrs = g.rowptr();
  vector<eidType> row_ptrs(rptrs, rptrs + g.V());
  row_ptrs.push_back(g.E());
  vidType* cptrs = g.colidx();
  vector<vidType> col_idxs(cptrs, cptrs + g.E());

  Graph sub_g;
  map<vidType, set<vidType>> parent_map;   // maps parent to children

  // create number of samples
  for (int s = 0; s < num_samples(); s++) {
    std::vector<vidType> inits = get_initial_transits(sample_size(-1), g.V());
    Sample sample_g(inits, &g);

    // continue sampling for defined number of steps
    for (int step = 0; step < steps(); step++) {
      vector<vidType> new_transits;
      if (sampling_type() == Collective) {
        for (int t_idx = 0; t_idx < sample_size(step); t_idx++) {
          int old_t_idx = t_idx % sample_size(step - 1);
          vector<vidType> old_t = sample_g.get_transits();
          vector<vidType> old_t_edges = sample_g.prev_edges(1, old_t_idx);
          // cout << "Old transit: " << old_t[0] << endl;
          // for (auto e: old_t_edges) cout << "Old transit edge: " << e << endl;
          vidType new_t = sample_next(&sample_g, old_t, old_t_edges, step);
          new_transits.push_back(new_t);
        }
      }
      else if (sampling_type() == Individual) {;
        // for (int t_idx = 0; t_idx < sample_size(step); t_idx++) {
        //   ;
        // }
      }
      sample_g.add_transits(new_transits);
    }
    for (auto p: sample_g.get_edges()) {
        parent_map[p.first].insert(p.second.begin(), p.second.end());
    }
  }
  cout << "Finished sampling" << endl;
  for (auto p: parent_map) cout << p.first << ": " << p.second.size() << endl;
  vidType nv = parent_map.size();
  eidType ne = 0;
  for (auto node: parent_map) { ne += node.second.size(); }
  sub_g.allocateFrom(nv, ne);

  vidType new_idx = 0;
  eidType offsets = 0;
  unordered_map<vidType, vidType> new_id_map;
  for (auto old_n: parent_map) { new_id_map[old_n.first] = new_idx++; }
  for (auto m: new_id_map) cout << m.first << " " << m.second << endl;
  cout << "EDGES: " << endl;
  for (auto old_n: parent_map) {
    offsets += old_n.second.size();
    sub_g.fixEndEdge(new_id_map[old_n.first], offsets);
    eidType offset = offsets - old_n.second.size();
    for (auto e: old_n.second) {
      cout << "edge num: " << offset << ", src: " << new_id_map[old_n.first] << ", dst:" << new_id_map[e] << endl;
      sub_g.constructEdge(offset++, new_id_map[e]);
    }
  }

  // for (auto m: new_id_map) cout << m.first << " " << m.second << endl;
  // cout << "nv: " << sub_g.num_vertices() << ", ne: " << sub_g.num_edges() << endl;
  // cout << "VERTICES:" << endl;
  // for (int i = 0; i <= sub_g.num_vertices(); i++) {
  //   cout << sub_g.rowptr()[i] << endl;
  // }
  // cout << "EDGES:" << endl;
  // for (int i = 0; i < sub_g.num_edges(); i++) {
  //   cout << sub_g.colidx()[i] << endl;
  // }
  return 0;
};