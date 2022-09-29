#include "graph.h"

//assumes symmetric graph
// 1) iterate over all remaining active vertices
// 2) for each active vertex, remove if induced degree < k. Any vertex removed has
//    core-number (k-1) (part of (k-1)-core, but not k-core)
// 3) stop once no vertices are removed. Vertices remaining are in the k-core.
void KCoreSolver(Graph &g, std::vector<int> &coreness, vidType &largest_core, int, int) {
  return;
}
