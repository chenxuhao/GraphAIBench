#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include <omp.h>
#include "sampling_utils.h"
#include "khop.h"
#include "samplegraph.h"
using namespace std;

void OMP_Sample(Graph &g) {
// CHECK FIXED RANDOMS
// void OMP_Sample(Graph &g, vector<vector<uint_fast32_t>> &random_nums, vector<vector<vidType>> &random_inits) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Graph Sampling (" << num_threads << " threads)\n";

  Graph sub_g;

  // create number of samples
  vector<vidType> inits = get_initial_transits(sample_size(-1) * num_samples(), g.V());
  // CHECK FIXED RANDOMS
  // std::vector<vidType> inits = random_inits[s];
  // for (auto init: inits) cout << "Sample " << s << " initial sample: " << init << endl;
  int step_count = sample_size(-1) * num_samples();
  int total_count = step_count;
  for (int step = 0; step < steps(); step++) {
    step_count *= sample_size(step);
    total_count += step_count;
  }
  Sample sample(total_count, &g);
  sample.add_inits(inits);

  Timer t;
  t.Start();
  // sample for defined number of steps
  step_count = sample_size(-1) * num_samples();
  int prev_step_count = num_samples();
  int t_begin = 0;
  int old_t_begin = 0;
  for (int step = 0; step < steps(); step++) {
    t_begin += step_count; 
    step_count *= sample_size(step);
    prev_step_count *= sample_size(step-1);
    if (sampling_type() == Individual) {
      // sample every new transit in the step for every sample group in parallel
      #pragma omp parallel for schedule(dynamic, 1)
      for (int idx = 0; idx < step_count; idx++) {
        int t_idx = t_begin + idx;
        int old_t_idx = old_t_begin + idx / sample_size(step);
        vidType old_t = sample.prev_vertex(1, old_t_idx);
        if (old_t == (numeric_limits<uint32_t>::max)()) {
          sample.write_transit(t_idx, (numeric_limits<uint32_t>::max)());
          continue;
        }
        vidType old_t_degree = sample.prev_vertex_degree(1, old_t);
        vidType new_t = (numeric_limits<uint32_t>::max)();
        if (old_t_degree != 0) { 
          new_t = sample_next(&sample, old_t, old_t_degree, step);
          // CHECK FIXED RANDOMS
          // uint_fast32_t rand_n = random_nums[step][idx];
          // new_t = sample_next_fixed(sample_g, old_t, old_t_edges, step, rand_n);
        }
        sample.write_transit(t_idx, new_t);
      }
    }
    else if (sampling_type() == Collective) {;
      // for (int t_idx = 0; t_idx < sample_size(step); t_idx++) {
      //   ;
      // }
    }
    old_t_begin += prev_step_count;
  }

  t.Stop();

  std::cout << "result size: " << step_count + t_begin << endl;

  // cout << "NEW SAMPLE~~~~~~~~~~~~~~~~~~" << endl;
  // vector<vidType> t_order = sample.get_all_transits();
  // cout << "[ ";
  // for (uint step = 0; step < t_order.size(); step++) {
  //   cout << t_order[step] << " ";
  // }
  // cout << "]" << endl;

  // map<vidType, set<vidType>> parent_map;   // maps parent to children
  // for (auto sample_g: samples) {
  //   int step_count = sample_size(-1);
  //   vector<vector<vidType>> t_order = sample_g.get_transits_order();
  //   for (int step = 0; step < steps(); step++) {
  //     step_count *= sample_size(step);
  //     for (int t_idx = 0; t_idx < step_count; t_idx++) {
  //       int old_t_idx = t_idx / sample_size(step);
  //       vidType parent = t_order[step][old_t_idx];
  //       vidType child = t_order[step+1][t_idx];
  //       if (parent == (numeric_limits<uint32_t>::max)() || child == (numeric_limits<uint32_t>::max)()) { continue; }
  //       if (parent == child) { continue; }
  //       parent_map[t_order[step][old_t_idx]].insert(t_order[step+1][t_idx]);
  //       if (!is_directed()) { parent_map[t_order[step+1][t_idx]].insert(t_order[step][old_t_idx]); }
  //     }
  //   }
  // }

  cout << "Finished sampling in " << t.Seconds() << " sec" << endl;
  // cout << "New sampled subgraph: |V| " << parent_map.size() << endl;
  // // for (auto p: parent_map) cout << p.first << ": " << p.second.size() << endl;
  // vidType nv = parent_map.size();
  // eidType ne = 0;
  // for (auto node: parent_map) { ne += node.second.size(); }
  // sub_g.allocateFrom(nv, ne);

  // vidType new_idx = 0;
  // eidType offsets = 0;
  // unordered_map<vidType, vidType> new_id_map;
  // for (auto old_n: parent_map) { new_id_map[old_n.first] = new_idx++; }
  // // for (auto m: new_id_map) cout << m.first << " " << m.second << endl;
  // // cout << "EDGES: " << endl;
  // for (auto old_n: parent_map) {
  //   offsets += old_n.second.size();
  //   sub_g.fixEndEdge(new_id_map[old_n.first], offsets);
  //   eidType offset = offsets - old_n.second.size();
  //   for (auto e: old_n.second) {
  //     // cout << "edge num: " << offset << ", src: " << new_id_map[old_n.first] << ", dst:" << new_id_map[e] << endl;
  //     sub_g.constructEdge(offset++, new_id_map[e]);
  //   }
  // }
  // cout << "New sampled subgraph: |V| " << sub_g.V() << " |E| " << sub_g.E() << endl;
  return;
};
