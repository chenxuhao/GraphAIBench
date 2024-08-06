#include <iostream>
#include <fstream>
#include "../../include/graph.h"
#include "../../include/deepwalk.h"
using namespace std;

#define SAMPLE_SIZE 1 // fanout of the random walk

// The DeepWalk algorithm generates a random walk \gamma times from each vertex, generating a total of \gamma * |V| walks.
// n_samples = 1 as of now but should change it to \gamma or |V| or \gamma * |V|? Because we don't want to call the function multiple times.

/**
 * @param g is a reference to the graph 
 * @param all_transits is where transits will be stored
 * @param initial_vertex is the first vertex of the walk
 * @param steps is the length of each random walk
*/
int sample(Graph &g, vector<vidType> &all_transits, vidType initial_vertex, int n_samples, int steps) {
  Timer t;
  t.Start();
  // all_transits keeps track of the sampled nodes at each step of the algorithm
  all_transits.push_back(initial_vertex);

  int step_count = SAMPLE_SIZE * n_samples; // step_count = 1
  int prev_step_count = n_samples; // prev_step_count = 1
  int t_begin = 0;
  int old_t_begin = 0;
  for (int step = 0; step < steps; step++) {
    t_begin += step_count; // t_begin++
    step_count *= SAMPLE_SIZE; // step_count = 1
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

    old_t_begin += prev_step_count; // old_t_begin++
  }

  t.Stop();

  std::cout << "result size: " << step_count + t_begin << endl;
  std::cout << "Finished sampling in " << t.Seconds() << " sec" << endl;

  return 0;
};

/**
int main(int argc, char * argv[]) { // argv = [my_program, prefix, n_samples, steps]
  Graph g;
  string prefix = argv[1];
  g.load_graph(prefix);
  cout << "Loading " << prefix << endl;
  g.print_meta_data();
  int n_samples = atoi(argv[2]);
  int steps = atoi(argv[3]);
  vector<vidType> all_transits({});
  RandomWalk(g, all_transits, 0, n_samples, steps);
  for (int i = 0; i < all_transits.size(); i++) {
    std::cout << all_transits[i] << endl;
  }
  return 0;
}
*/
