#include <iostream>
#include <fstream>
#include <omp.h>
#include "../../include/graph.h"
#include "../../include/deepwalk.h"
using namespace std;

#define SAMPLE_SIZE 1 // fanout of the random walk
#define GAMMA 2

// The DeepWalk algorithm generates a random walk gamma times from each vertex, generating a total of gamma * |V| walks.
// N = gamma * |V| and n_samples = |V|? Because we don't want to call the function multiple times.

/**
 * @param g is a reference to the graph 
 * @param all_transits is where transits will be stored
 * @param n_samples is the number of different initial vertices in the sample (|V| in the case of DeepWalk)
 * @param n_threads is the number of threads
*/
// int sample(Graph &g, vector<vidType> &all_transits, vidType initial_vertex, int n_samples, int steps) {
int sample_alg(Graph &g, vector<vidType> &all_transits, int n_samples, int n_threads, int total_count=0) {
  int num_threads = 1;
  omp_set_num_threads(n_threads);
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Graph Sampling (" << num_threads << " threads)\n";

  Timer t;
  t.Start();
  int N = n_samples * GAMMA; // total number of initial samples = gamma * |V|
  all_transits.clear();
  // all_transits keeps track of the sampled nodes at each step of the algorithm
  vector<vidType> all_vertices = get_initial_transits(g.V(), g.V());
  for (int i = 0; i < g.V(); i++) {
    for (int j = 0; j < GAMMA; j++) {
      all_transits.push_back(all_vertices[i]);
    }
  }

  int step_count = SAMPLE_SIZE * N; // step_count = n_samples = gamma * |V|
  int prev_step_count = N; // prev_step_count = n_samples
  int t_begin = 0;
  int old_t_begin = 0;
  for (int step = 0; step < steps(); step++) { // each step of the random walk
    t_begin += step_count; // t_begin += n_samples
    step_count *= SAMPLE_SIZE; // step_count = n_samples (does not change)
    prev_step_count *= SAMPLE_SIZE; // prev_step_count = n_samples (does not change)

    // assume individual transit sampling - otherwise, we would consider whether samplingType() is Individual or Collective
    #pragma omp parallel for
    for (int idx = 0; idx < step_count; idx++) { // idx = 0 -> n_samples - 1
      int t_idx = t_begin + idx;
      int old_t_idx = old_t_begin + idx / SAMPLE_SIZE; // old_t_idx = old_t_begin + idx
      vidType old_t = all_transits[old_t_idx]; // old_t = previous transit vertex
      if (old_t == (numeric_limits<uint32_t>::max)()) { 
        all_transits[t_idx] = (numeric_limits<uint32_t>::max)();
        continue;
      }
      vidType old_t_degree = g.out_degree(old_t);
      vidType new_t = (numeric_limits<uint32_t>::max)();
      if (old_t_degree != 0) { 
        new_t = sample_next(g, old_t, old_t_degree, step);
      }
      all_transits[t_idx] = new_t; // the new node (new_t) is stored in all_transits at index t_idx
    }

    old_t_begin += prev_step_count; // old_t_begin += n_samples
  }

  t.Stop();

  std::cout << "result size: " << step_count + t_begin << endl;
  std::cout << "Finished sampling in " << t.Seconds() << " sec" << endl;

  return 0;
};