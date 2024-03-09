#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "sampling_utils.h"
#include "node2vec.h"
#include "samplegraph.h"
using namespace std;

void OMP_Sample(Graph &g);
// void CILK_Sample(Graph &g);
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
  vector<Sample> samples;

  // create number of samples
  for (int s = 0; s < num_samples(); s++) {
    vector<vidType> inits = get_initial_transits(sample_size(-1), g.V());
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
          new_t = sample_next(sample_g, old_t, old_t_edges, step);
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

  OMP_Sample(g);
  // CILK_Sample(g);
  return 0;
};
