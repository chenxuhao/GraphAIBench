#include "graph.h"

int main(int argc,char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph\n";
    abort();
  }
  OutOfCoreGraph g(argv[1]);
  g.print_meta_data();
  int v = atoi(argv[2]);
  auto deg = g.get_degree(v);
  std::cout << "vertex " << v << " degree " << deg << "\n";
  //#pragma omp parallel for
  //for (vidType v = 0; v < g.V(); v++) {
  for (int i = 0; i < 10; i++) {
    std::cout << "neighbor[" << i << "]=" << g.N(v, i) << "\n";
  }
  return 0;
}
