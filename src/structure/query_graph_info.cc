#include "graph.h"

int main(int argc,char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> <vertex_id> [bin_width(100)]\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 100\n";
    abort();
  }
  OutOfCoreGraph g(argv[1]);
  g.print_meta_data();
  int bin_width = 0;
  int v = 0;
  v = atoi(argv[2]);
  auto deg = g.get_degree(v);
  std::cout << "vertex " << v << " degree " << deg << "\n";
  if (argc > 2) bin_width = atoi(argv[3]);
  if (bin_width) g.degree_histogram(bin_width);
  //#pragma omp parallel for
  //for (vidType v = 0; v < g.V(); v++) {
  //for (int i = 0; i < 10; i++) {
  //  std::cout << "neighbor[" << i << "]=" << g.N(v, i) << "\n";
  //}
  return 0;
}
