#include "graph.h"
#include "compressor.hh"
#include "khop.h"


void kHopSolver(Graph &g, int n_samples, int n_threads) {
  vector<vidType> inits = get_initial_transits(sample_size(-1) * n_samples, g.V());
  int step_count = sample_size(-1) * n_samples;
  int total_count = step_count;
  for (int step = 0; step < steps(); step++) {
    step_count *= sample_size(step);
    total_count += step_count;
  }
  std::vector<vidType> transits(total_count, 0);
  for (int i = 0; i < inits.size(); i++) {
    transits[i] = inits[i];
  }
  std::cout << "...initialized starting transits..." << std::endl;

  Timer t;
  t.Start();
  // sample for defined number of steps
  step_count = sample_size(-1) * n_samples;
  int prev_step_count = n_samples;
  int t_begin = 0;
  int old_t_begin = 0;
  for (int step = 0; step < steps(); step++) {
    std::cout << "STEP " << step << std::endl;
    t_begin += step_count;
    step_count *= sample_size(step);
    prev_step_count *= sample_size(step-1);
    // sample every new transit in the step for every sample group
    for (int idx = 0; idx < step_count; idx++) {
      int t_idx = t_begin + idx;
      int old_t_idx = old_t_begin + idx / sample_size(step);
      vidType old_t = transits[old_t_idx];
      if (old_t == (numeric_limits<uint32_t>::max)()) {
        transits[t_idx] = (numeric_limits<uint32_t>::max)();
        continue;
      }
      vidType new_t = sample_next_vbyte(g, old_t);
      transits[t_idx] = new_t;
    }
    old_t_begin += prev_step_count;
  }
  t.Stop();
  std::cout << "result size: " << step_count + t_begin << std::endl;
  std::cout << "Finished sampling in " << t.Seconds() << " sec" << std::endl;
}


int main(int argc, char* argv[]) {
  Graph g;
  std::string in_prefix = argv[1];
  std::string out_prefix = argv[2];
  std::string scheme = "streamvbyte";
  bool permutated = false;
  save_compressed_graph(in_prefix, out_prefix);
  g.load_compressed_graph(out_prefix, scheme, permutated);
  g.print_meta_data();
  std::cout << "LOADED COMPRESSED GRAPH\n" << std::endl;

  // std::cout << "Begin sampling compressed graph..." << std::endl;
  // int n_samples = argc >= 4 ? atoi(argv[3]) : num_samples();
  // int n_threads = argc >= 5 ? atoi(argv[4]) : 1;
  // kHopSolver(g, n_samples, n_threads);
  return 0;
}