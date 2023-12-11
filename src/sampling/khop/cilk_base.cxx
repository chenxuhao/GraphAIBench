#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "cilk.h"
#include "cilk_api.h"
#include "sampling_utils.h"
#include "khop.h"
#include "samplegraph.h"
using namespace std;

void CILK_Sample(Graph &g, int n_samples) {
// CHECK FIXED RANDOMS
// void CILK_Sample(Graph &g, vector<vector<uint_fast32_t>> &random_nums, vector<vector<vidType>> &random_inits) {
  int num_threads = __cilkrts_get_nworkers();
  std::cout << "Cilk Graph Sampling (" << num_threads << " threads)\n";

  Graph sub_g;

  // create number of samples
  vector<vidType> inits = get_initial_transits(sample_size(-1) * n_samples, g.V());
  // CHECK FIXED RANDOMS
  // random_inits.push_back(inits);
  // for (auto init: inits) cout << "Sample " << s << " initial sample: " << init << endl;
  int step_count = sample_size(-1) * n_samples;
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
  step_count = sample_size(-1) * n_samples;
  int prev_step_count = n_samples;
  int t_begin = 0;
  int old_t_begin = 0;
  for (int step = 0; step < steps(); step++) {
    t_begin += step_count;
    step_count *= sample_size(step);
    prev_step_count *= sample_size(step-1);
    if (sampling_type() == Individual) {
      // sample every new transit in the step for every sample group
      // CHECK FIXED RANDOMS
      // allocate_transits(random_nums, step_count * n_samples);
      #pragma grainsize = 32
      cilk_for (int idx = 0; idx < step_count; idx++) {
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
          new_t = sample_next(sample, old_t, old_t_degree, step);
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
  cout << "Finished sampling in " << t.Seconds() << " sec" << endl;

  return;
};
