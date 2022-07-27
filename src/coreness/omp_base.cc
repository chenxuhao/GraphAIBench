// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "platform_atomics.h"

//assumes symmetric graph
// 1) iterate over all remaining active vertices
// 2) for each active vertex, remove if induced degree < k. Any vertex removed has
//    core-number (k-1) (part of (k-1)-core, but not k-core)
// 3) stop once no vertices are removed. Vertices remaining are in the k-core.
void KCoreSolver(Graph &g, std::vector<int> &coreness, vidType &largest_core, int, int) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  auto nv = g.V();
  std::vector<int8_t> toRemove(nv, 0);
  std::vector<int> degrees(nv); // induced degree; inactive vertex if degree = -1
  for (vidType u = 0; u < nv; u ++)
    degrees[u] = g.get_degree(u);
  largest_core = -1;
  vidType total_num_removed = 0;
  std::cout << "OpenMP k-core decomposition (" << num_threads << " threads)\n";

  Timer t;
  t.Start();
  for (vidType k = 1; k <= nv; k++) {
    while (1) {
      vidType num_toRemove = 0;
      #pragma omp parallel for reduction(+:num_toRemove)
      for (vidType u = 0; u < nv; u ++) {
        if (degrees[u] != -1 && degrees[u] < k) {
          coreness[u] = k-1;
          degrees[u] = -1;
          toRemove[u] = 1;
          num_toRemove += 1;
        }
      }
      if (num_toRemove == 0) {
        std::fill(toRemove.begin(), toRemove.end(), 0);
        break;
      } else {
        //std::cout << "Removing " << num_toRemove << " " << k-1  << "-core vertices\n";
        // remove the vertices and their edges
        #pragma omp parallel for
        for (vidType v = 0; v < nv; v ++) {
          if (toRemove[v] == 1) {
            for (auto u : g.N(v)) {
              if (degrees[u] > 0)
                //degrees[u]--;
                fetch_and_add(degrees[u], -1);
            }
          }
        }
        total_num_removed += num_toRemove;
        std::fill(toRemove.begin(), toRemove.end(), 0);
      }
    }
    //std::cout << total_num_removed << " vertices removed so far\n";
    if (total_num_removed == nv) { largest_core = k-1; break; }
  }
  t.Stop();
  std::cout << "runtime [kcore_omp_base] = " << t.Seconds() << " sec\n";
  return;
}

