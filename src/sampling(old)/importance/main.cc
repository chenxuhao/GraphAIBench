#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "sampling_utils.h"
#include "importance_sampling.h"
#include "samplegraph.h"
using namespace std;


// void OMP_Sample(Graph &g);
// void CILK_Sample(Graph &g);
// CHECK FIXED RANDOMS
void OMP_Sample(Graph &g, vector<vector<uint_fast32_t>> &random_nums, vector<vector<vidType>> &random_inits);
// void CILK_Sample(Graph &g, vector<vector<uint_fast32_t>> &random_nums, vector<vector<vidType>> &random_inits);
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

  // CHECK FIXED RANDOMS
  vector<vector<uint_fast32_t>> random_nums;
  vector<vector<vidType>> random_inits;

  Graph sub_g;
  vector<Sample> samples;

  // create number of samples
  for (int s = 0; s < num_samples(); s++) {
    vector<vidType> inits = get_initial_transits(sample_size(-1), g.V());
    // CHECK FIXED RANDOMS
    random_inits.push_back(inits);
    // for (auto init: inits) cout << "Sample " << s << " initial sample: " << init << endl;
    Sample sample(inits, &g);
    for (int step = 0; step < steps(); step++) {
      sample.allocate_layer(sample_size(step));
    }
    samples.push_back(sample);
  }

  Timer t;
  t.Start();
  // sample for defined number of steps
  for (int step = 0; step < steps(); step++) {
    if (sampling_type() == Individual) {
        //;
    }
    else if (sampling_type() == Collective) {;
      // CHECK FIXED RANDOMS
      allocate_transits(random_nums, sample_size(step) * num_samples());
      for (int idx = 0; idx < sample_size(step) * num_samples(); idx++) {
        int t_idx = idx % sample_size(step);
        Sample* sample_g = &samples[idx / sample_size(step)]; 
        vector<vidType> old_t = sample_g->get_transits();
        vector<vidType> old_t_edges = {};
        // vidType new_t = sample_next(sample_g, old_t, old_t_edges, step);
        // CHECK FIXED RANDOMS
        uint_fast32_t rand_idx;
        vidType new_t;
        tie(new_t, rand_idx) = sample_next_store(sample_g, old_t, old_t_edges, step);
        random_nums[step][idx] = rand_idx;
        sample_g->write_transit(t_idx, new_t);
      }
    }
    for (auto& s: samples) { 
      s.increment_filled_layer();
    }
  }
  t.Stop();

  for (auto& s: samples) {
    cout << "NEW SAMPLE~~~~~~~~~~~~~~~~~~" << endl;
    vector<vector<vidType>> t_order = s.get_transits_order();
    for (uint step = 0; step < t_order.size(); step++) {
      cout << "[ ";
      vector<vidType> layer = t_order[step];
      for (uint l = 0; l < layer.size(); l++) {
        cout << layer[l] << " ";
      }
      cout << "]" << endl;
    }
  }

  map<vidType, set<vidType>> parent_map;   // maps parent to children
  for (auto sample_g: samples) {
    vector<vector<vidType>> t_order = sample_g.get_transits_order();
    for (int step = 0; step < steps(); step++) {
      for (auto child: t_order[step+1]) {
        for (auto parent: t_order[step]) {
          if (parent == child) { continue; }
          if (sample_g.get_graph()->is_connected(child, parent)) {
            parent_map[parent].insert(child);
            if (!is_directed()) { parent_map[child].insert(parent); }
          }
        }
      }
    }
  }

  cout << "Finished sampling in " << t.Seconds() << " sec" << endl;
  // for (auto p: parent_map) cout << p.first << ": " << p.second.size() << endl;
  vidType nv = parent_map.size();
  eidType ne = 0;
  for (auto node: parent_map) { ne += node.second.size(); }
  sub_g.allocateFrom(nv, ne);

  vidType new_idx = 0;
  eidType offsets = 0;
  unordered_map<vidType, vidType> new_id_map;
  for (auto old_n: parent_map) { new_id_map[old_n.first] = new_idx++; }
  for (auto old_n: parent_map) {
    offsets += old_n.second.size();
    sub_g.fixEndEdge(new_id_map[old_n.first], offsets);
    eidType offset = offsets - old_n.second.size();
    for (auto e: old_n.second) {
      // cout << "edge num: " << offset << ", src: " << new_id_map[old_n.first] << ", dst:" << new_id_map[e] << endl;
      sub_g.constructEdge(offset++, new_id_map[e]);
    }
  }
  cout << "New sampled subgraph: |V| " << sub_g.V() << " |E| " << sub_g.E() << endl;
  // cout << "old idx to new idx mapping" << endl;
  // cout << "{ ";
  // for (auto m: new_id_map) cout << m.first << ": " << m.second << ", ";
  // cout << "}" << endl;
  // cout << "nv: " << sub_g.num_vertices() << ", ne: " << sub_g.num_edges() << endl;
  // cout << "rowptrs: [ ";
  // for (int i = 0; i <= sub_g.num_vertices(); i++) {
  //   cout << sub_g.rowptr()[i] << " ";
  // }
  // cout << "]" << endl;
  // cout << "colptrs: [ ";
  // for (int i = 0; i < sub_g.num_edges(); i++) {
  //   cout << sub_g.colidx()[i] << " ";
  // }
  // cout << "]" << endl;

  // OMP_Sample(g);
  // CILK_Sample(g);
  // CHECK FIXED RANDOMS
  OMP_Sample(g, random_nums, random_inits);
  // CILK_Sample(g, random_nums, random_inits);
  return 0;
};
