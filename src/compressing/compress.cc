#include "graph.h"
#include "compressor.hh"
#include "khop.h"


void save_compressed_graph(std::string in_prefix, std::string out_prefix) {
  int permutate = 0, degree_threshold = 32;
  int alignment = 2; // 0: not aligned; 1: byte aligned; 2: word aligned
  bool reverse = false; // reverse hybrid scheme: low-degree vbyte; high-degree unary
  bool use_unary = false;
  std::string scheme = "streamvbyte";

  GraphTy g(in_prefix);
  g.print_meta_data();

  bool pre_encode = g.V() > 1000000;
  unary_encoder *encoder = NULL;
  Compressor compressor(scheme, out_prefix, use_unary, &g, encoder, permutate, degree_threshold, alignment);
  std::cout << "start compression ...\n";
  compressor.compress(pre_encode, reverse);
  compressor.print_stats();
  std::cout << "writing compressed graph to disk ...\n";
  compressor.write_compressed_graph();
  std::cout << "compression completed!\n";
  copy_meta_file(in_prefix, out_prefix);
  std::cout << "meta file copied over\n";
}


void kHopSolver(Graph &g, int n_samples, int n_threads) {
  // create graph and retrieve node/edge data
  eidType* rptrs = g.rowptr();
  vector<eidType> row_ptrs(rptrs, rptrs + g.V());
  row_ptrs.push_back(g.E());
  vidType* cptrs = g.colidx();
  vector<vidType> col_idxs(cptrs, cptrs + g.E());

  Graph sub_g;

  // create number of samples
  vector<vidType> inits = get_initial_transits(sample_size(-1) * n_samples, g.V());
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
    // sample every new transit in the step for every sample group
    for (int idx = 0; idx < step_count; idx++) {
      int t_idx = t_begin + idx;
      int old_t_idx = old_t_begin + idx / sample_size(step);
      // cout << "sample idx: " << idx / step_count << ", t_idx: " << t_idx << ", old_t_idx: " << old_t_idx << endl;
      vidType old_t = sample.prev_vertex(1, old_t_idx);
      if (old_t == (numeric_limits<uint32_t>::max)()) {
        sample.write_transit(t_idx, (numeric_limits<uint32_t>::max)());
        continue;
      }
      vidType old_t_degree = sample.prev_vertex_degree(1, old_t);
      vidType new_t = (numeric_limits<uint32_t>::max)();
      if (old_t_degree != 0) { 
        new_t = sample_next_vbyte(sample, old_t);
      }
      sample.write_transit(t_idx, new_t);
    }
    old_t_begin += prev_step_count;
  }
  t.Stop();
  std::cout << "result size: " << step_count + t_begin << endl;
  std::cout << "Finished sampling in " << t.Seconds() << " sec" << endl;
}


int main(int argc, char* argv[]) {
  Graph g;
  std::string in_prefix = argv[1];
  std::string out_prefix = argv[2];
  // save_compressed_graph(in_prefix, out_prefix);
  g.load_compressed_graph(out_prefix, "streamvbyte", false);
  g.print_meta_data();
  std::cout << "SAVED AND LOADED" << std::endl;

  int n_samples = argc >= 3 ? atoi(argv[3]) : num_samples();
  int n_threads = argc >= 4 ? atoi(argv[4]) : 1;
  kHopSolver(g, n_samples, n_threads);
  return 0;
}