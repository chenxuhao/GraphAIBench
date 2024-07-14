#include <iostream>
#include <fstream>
#include "graph.h"
#include "deepwalk.h"
using namespace std;

#define SAMPLE_SIZE 1 // fanout of the random walk

// The DeepWalk algorithm runs RandomWalk \gamma times from each vertex, generating a total of \gamma * |V| walks.

/**
 * @param g is a reference to the graph 
 * @param initial_vertex is the first vertex of the walk
 * @param steps is the length of each random walk
*/
int RandomWalk(Graph &g, vidType initial_vertex, int n_samples=1, int steps) {
  Timer t;
  t.Start();
  // all_transits keeps track of the sampled nodes at each step of the algorithm
  vector<vidType> all_transits = {initial_vertex};

  int step_count = SAMPLE_SIZE * n_samples; // step_count = prev_step_count = 1
  int prev_step_count = n_samples;
  int t_begin = 0;
  int old_t_begin = 0;
  for (int step = 0; step < steps; step++) {
    t_begin += step_count; // t_begin++
    step_count *= SAMPLE_SIZE;
    prev_step_count *= SAMPLE_SIZE; // step_count and prev_step_count do not change

    // assume individual transit sampling - otherwise, we would consider whether samplingType() is Individual or Collective
    for (int idx = 0; idx < step_count; idx++) { // idx = 0 is the only option if step_count = 1
      int t_idx = t_begin + idx; // t_idx = t_begin
      int old_t_idx = old_t_begin + idx / SAMPLE_SIZE; // old_t_idx = old_t_begin
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

    old_t_begin += prev_step_count;
  }

  t.Stop();

  std::cout << "result size: " << step_count + t_begin << endl;
  std::cout << "Finished sampling in " << t.Seconds() << " sec" << endl;

  return 0;
};