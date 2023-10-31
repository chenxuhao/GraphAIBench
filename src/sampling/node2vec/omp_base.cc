#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "sampling_utils.h"
#include "node2vec.h"
#include "samplegraph.h"
using namespace std;


void OMP_Sample(Graph &g) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Graph Sampling (" << num_threads << " threads)\n";

  Graph sub_g;
  map<vidType, set<vidType>> parent_map;   // maps parent to children

  omp_lock_t parentlock;

  omp_init_lock(&parentlock);

  Timer t;
  t.Start();
  #pragma omp parallel for schedule(dynamic, 1)
  for (int s = 0; s < num_samples(); s++) {
    std::vector<vidType> inits = get_initial_transits(sample_size(-1), g.V());
    Sample sample_g(inits, &g);

    // continue sampling for defined number of steps
    for (int step = 0; step < steps(); step++) {
      vector<vidType> new_transits;
      if (sampling_type() == Individual) {
        for (int t_idx = 0; t_idx < sample_size(step) * sample_size(step-1); t_idx++) {
          int old_t_idx = t_idx % sample_size(step-1);
          vector<vidType> old_t = {sample_g.prev_vertex(1, old_t_idx)};
          vector<vidType> old_t_edges = sample_g.prev_edges(1, old_t_idx);
          vidType new_t = sample_next(&sample_g, old_t, old_t_edges, step);
          new_transits.push_back(new_t);
          omp_set_lock(&parentlock);
          parent_map[old_t[0]].insert(new_t);
          if (!is_directed()) { parent_map[new_t].insert(old_t[0]); }
          omp_unset_lock(&parentlock);
        }
      }
      else if (sampling_type() == Collective) {;
        // for (int t_idx = 0; t_idx < sample_size(step); t_idx++) {
        //   ;
        // }
      }
      sample_g.add_transits(new_transits);
    }
  }
  t.Stop();
  omp_destroy_lock(&parentlock);

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
  return;
};
