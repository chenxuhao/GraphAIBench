#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <string>
#include <unordered_set>
#include "platform_atomics.h"
#include "sampling_utils.h"
#include "khop.h"
using namespace std;

int sample_alg(Graph &g, vector<vidType> &all_transits, int n_samples, int n_threads, int total_count=0) {
  int num_threads = 1;
  omp_set_num_threads(n_threads);
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Graph Sampling (" << num_threads << " threads)\n";

  // total_count -= n_samples * sample_size(-1);
  vector<uint_fast32_t> random_idxs(total_count);
  for (int i = 0; i < total_count; i++) {
    random_idxs[i] = gen();
  }


  vector<vidType> v_degs(g.V(), 0);
  #pragma omp parallel for
  for (vidType v = 0; v < g.V(); v++) {
    v_degs[v] = g.get_degree(v);
  }
  vector<vidType> v_accesses(g.V(), 0);

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
    #pragma omp parallel for
    for (int idx = 0; idx < prev_step_count; idx++) {
      int old_t_idx = old_t_begin + idx;
      unordered_set<vidType> exists;
      for (int j = 0; j < sample_size(step); j++) {
        int t_idx = t_begin + idx * sample_size(step) + j;
        vidType old_t = all_transits[old_t_idx];
        if (old_t == (numeric_limits<uint32_t>::max)()) {
          all_transits[t_idx] = (numeric_limits<uint32_t>::max)();
          continue;
        }
        vidType old_t_degree = g.out_degree(old_t);
        vidType new_t = (numeric_limits<uint32_t>::max)();
        if (old_t_degree != 0) { 
          uint_fast32_t random_idx = random_idxs[t_idx];
          new_t = sample_next(g, old_t, old_t_degree, step, random_idx);
          // std::cout << "old_t " << old_t << " idx " << random_idx << " new_t " << new_t << " deg " << old_t_degree << std::endl;
          // fetch_and_add(v_accesses[new_t], 1);
          if (exists.find(new_t) == exists.end()) {
            fetch_and_add(v_accesses[new_t], 1);
            exists.insert(new_t);
          }
        }
        all_transits[t_idx] = new_t;
      }
    }
    old_t_begin += prev_step_count;
  }

  t.Stop();

  std::ofstream out_file("/home/mcai1/access_orkut.txt");
  std::ofstream out_file2("/home/mcai1/access_orkut2.txt");
  vidType second = g.V() / 5;
  vidType count = 0;
  for (vidType v = 0; v < second; v++) {
    if (v_accesses[v] == 0) continue;
    if (count % 10 == 0 || count < 100) {
      std::string line = std::to_string(v_degs[v]) + "," + std::to_string(v_accesses[v]) + "\n";
      out_file << line;
    }
    count++;
  }
  out_file.close();
  for (vidType v = second; v < g.V(); v++) {
    if (v_accesses[v] == 0) continue;
    if (count % 10 == 0) {
    std::string line = std::to_string(v_degs[v]) + "," + std::to_string(v_accesses[v]) + "\n";
    out_file2 << line;
    }
    count++;
  }
  out_file2.close();

  std::cout << "result size: " << step_count + t_begin << endl;
  std::cout << "Finished sampling in " << t.Seconds() << " sec" << endl;

  return 0;
};
