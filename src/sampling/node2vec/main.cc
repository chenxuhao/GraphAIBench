#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "sampling_utils.h"
#include "node2vec.h"
#include "samplegraph.h"
using namespace std;

void OMP_Sample(Graph &g, vector<vector<uint_fast32_t>> &random_nums, vector<vector<uint_fast32_t>> &random_inits);
// void CILK_Sample(Graph &g, vector<vector<uint_fast32_t>> &random_nums, vector<vector<uint_fast32_t>> &random_inits);
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

  vector<vector<uint_fast32_t>> random_nums = generate_randoms();
  vector<vector<uint_fast32_t>> random_inits = generate_init_randoms();
  Graph sub_g;
  vector<Sample> samples;

  vector<vector<uint_fast32_t>> random_nums;
  vector<vector<vidType>> random_inits;

  // create number of samples
  for (int s = 0; s < num_samples(); s++) {
    std::vector<vidType> inits = get_initial_transits(sample_size(-1), g.V(), random_inits[s]);
    // for (auto init: inits) cout << "Sample " << s << " initial sample: " << init << endl;
    Sample sample(inits, &g);
    int step_count = sample_size(-1);
    for (int step = 0; step < steps(); step++) {
      step_count *= sample_size(step);
      sample.allocate_layer(step_count);
    }
    samples.push_back(sample);
  }

  Timer t;
  t.Start();
  // sample for defined number of steps
  int step_count = sample_size(-1);
  for (int step = 0; step < steps(); step++) {
    step_count *= sample_size(step);
    if (sampling_type() == Individual) {
      // sample every new transit in the step for every sample group
      for (int idx = 0; idx < step_count * num_samples(); idx++) {
        int t_idx = idx % step_count;
        Sample* sample_g = &samples[idx / step_count]; 
        int old_t_idx = t_idx / sample_size(step);
        vector<vidType> old_t = {sample_g->prev_vertex(1, old_t_idx)};
        if (old_t[0] == (numeric_limits<uint32_t>::max)()) {
          sample_g->write_transit(t_idx, (numeric_limits<uint32_t>::max)());
          continue;
        }
        vector<vidType> old_t_edges = sample_g->prev_edges(1, old_t_idx);
        vidType new_t = (numeric_limits<uint32_t>::max)();
        if (old_t_edges.size() != 0) { 
          uint_fast32_t rand_n = random_nums[step][idx];
          new_t = sample_next(sample_g, old_t, old_t_edges, step, rand_n);
        }
        sample_g->write_transit(t_idx, new_t);
      }
    }
    else if (sampling_type() == Collective) {;
      // for (int t_idx = 0; t_idx < sample_size(step); t_idx++) {
      //   ;
      // }
    }
    for (auto& s: samples) { 
      s.increment_filled_layer();
    }
  }
  t.Stop();

  // for (auto& s: samples) {
  //   cout << "NEW SAMPLE~~~~~~~~~~~~~~~~~~" << endl;
  //   vector<vector<vidType>> t_order = s.get_transits_order();
  //   for (uint step = 0; step < t_order.size(); step++) {
  //     cout << "[ ";
  //     vector<vidType> layer = t_order[step];
  //     for (uint l = 0; l < layer.size(); l++) {
  //       cout << layer[l] << " ";
  //     }
  //     cout << "]" << endl;
  //   }
  // }

  map<vidType, set<vidType>> parent_map;   // maps parent to children
  for (auto sample_g: samples) {
    int step_count = sample_size(-1);
    vector<vector<vidType>> t_order = sample_g.get_transits_order();
    for (int step = 0; step < steps(); step++) {
      step_count *= sample_size(step);
      for (int t_idx = 0; t_idx < step_count; t_idx++) {
        int old_t_idx = t_idx / sample_size(step);
        vidType parent = t_order[step][old_t_idx];
        vidType child = t_order[step+1][t_idx];
        if (parent == (numeric_limits<uint32_t>::max)() || child == (numeric_limits<uint32_t>::max)()) { continue; }
        if (parent == child) { continue; }
        parent_map[t_order[step][old_t_idx]].insert(t_order[step+1][t_idx]);
        if (!is_directed()) { parent_map[t_order[step+1][t_idx]].insert(t_order[step][old_t_idx]); }
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
  // for (auto m: new_id_map) cout << m.first << " " << m.second << endl;
  // cout << "EDGES: " << endl;
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

  OMP_Sample(g);
  // CILK_Sample(g);
  return 0;
};
