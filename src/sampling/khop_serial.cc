#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "sampling_utils.h"
#include "khop.h"
using namespace std;

int sample_alg(Graph &g, vector<vidType> &all_transits, int n_samples, int n_threads) {
  Timer t;
  t.Start();
  int step_count = sample_size(-1) * n_samples;
  int prev_step_count = n_samples;
  int t_begin = 0;
  int old_t_begin = 0;
  for (int step = 0; step < steps(); step++) {
    t_begin += step_count;
    step_count *= sample_size(step);
    prev_step_count *= sample_size(step-1);
    if (sampling_type() == Individual) {;
      for (int idx = 0; idx < step_count; idx++) {
        int t_idx = t_begin + idx;
        int old_t_idx = old_t_begin + idx / sample_size(step);
        // cout << "sample idx: " << idx / step_count << ", t_idx: " << t_idx << ", old_t_idx: " << old_t_idx << endl;
        vidType old_t = all_transits[old_t_idx];
        if (old_t == (numeric_limits<uint32_t>::max)()) {
          all_transits[t_idx] = (numeric_limits<uint32_t>::max)();
          continue;
        }
        vidType old_t_degree = g.out_degree(old_t);
        vidType new_t = (numeric_limits<uint32_t>::max)();
        if (old_t_degree != 0) { 
          new_t = sample_next(g, old_t, old_t_degree, step);
        }
        all_transits[t_idx] = new_t;
      }
    }
    else if (sampling_type() == Collective) {;
    // ignore for now, assume individual sampling
      // for (int t_idx = 0; t_idx < sample_size(step); t_idx++) {
      //   ;
      // }
    }
    old_t_begin += prev_step_count;
  }

  t.Stop();

  std::cout << "result size: " << step_count + t_begin << endl;
  std::cout << "Finished sampling in " << t.Seconds() << " sec" << endl;

  return 0;
};
