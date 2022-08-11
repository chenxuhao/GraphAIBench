#include "graph.h"

int main(int argc,char *argv[]) {
  if (argc != 2) {
    std::cout << "incorrect arguments." << std::endl;
    std::cout << "<input_path>" << std::endl;
    abort();
  }
  Graph g;
  g.load_compressed_graph(argv[1]);
  g.print_meta_data();
  g.decompress();
  g.print_graph();

  //uint64_t total = 0;
  //tc_compressed(g, total);
  //std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}
