#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
// #include <cilk/cilk.h>
// #include <cilk/cilk_api.h>
// #include <omp.h>
#include "sampling_utils.h"
#include "node2vec.h"
#include "samplegraph.h"
using namespace std;


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

  // int world_rank, world_size;

  // MPI_Init(NULL, NULL); // initialize MPI library

  // MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get my process id

  // int num_tasks = num_samples();
  // int ntasks_per_rank = num_tasks / world_size + (num_tasks % world_size != 0); // ceiling division
  // int begin = ntasks_per_rank * world_rank;
  // int end = ntasks_per_rank * (world_rank+1);
  // if (end > num_tasks) end = num_tasks;

  // create number of samples
  for (int s = 0; s < num_samples(); s++) {
    std::vector<vidType> inits = get_initial_transits(sample_size(-1), g.V());
    for (auto init: inits) cout << "Sample " << s << " initial sample: " << init << endl;
    Sample sample_g(inits, &g);

    // continue sampling for defined number of steps
    for (int step = 0; step < steps(); step++) {
      vector<vidType> new_transits;
      if (sampling_type() == Individual) {
        for (int t_idx = 0; t_idx < sample_size(step) * sample_size(step-1); t_idx++) {
          int old_t_idx = t_idx % sample_size(step-1);
          vector<vidType> old_t = {sample_g.prev_vertex(1, old_t_idx)};
          vector<vidType> old_t_edges = sample_g.prev_edges(1, old_t_idx);
          // cout << "Old transit: " << old_t[0] << endl;
          // for (auto e: old_t_edges) cout << "Old transit edge: " << e << endl;
          vidType new_t = sample_next(&sample_g, old_t, old_t_edges, step);
          new_transits.push_back(new_t);
          parent_map[old_t[0]].insert(new_t);
          if (!is_directed()) { parent_map[new_t].insert(old_t[0]); }
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

  // MPI_Finalize();

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
