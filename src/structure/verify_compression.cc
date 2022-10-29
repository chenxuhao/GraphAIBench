#include "graph.h"

int main(int argc,char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> <compressed_graph>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph ../../inputs/mico/cgr\n";
    abort();
  }
  OutOfCoreGraph g(argv[1]);
  g.print_meta_data();
  Graph cg;
  cg.load_compressed_graph(argv[2]);
  cg.print_meta_data();
  assert(g.V()==cg.V());
  assert(g.E()==cg.E());
  assert(g.get_max_degree()==cg.get_max_degree());
  //cg.decompress();
  std::cout << "Start comparison\n";
  //#pragma omp parallel for
  for (vidType v = 0; v < g.V(); v++) {
    VertexSet adj_v(v); // u's adjacency list
    cg.decode_vertex(v, adj_v, true);
    auto deg = g.get_degree(v);
    auto c_deg = adj_v.size();
    if (deg != c_deg) {
      std::cout << "g(" << v << ") deg " << deg << " ";
      std::cout << "cg(" << v << ") deg " << c_deg << "\n";
    }
    if (v == 3701) {
      std::cout << "cg(" << v << ") deg " << c_deg << "\n";
      for (vidType i = 0; i < deg; i ++) {
        if (i < 9 || g.N(v, i) != adj_v[i])
          std::cout << "cg.N(" << v << "," << i << ") = " << adj_v[i] << "\n";
        if (g.N(v, i) != adj_v[i]) {
          std::cout << "g.N(" << v << "," << i << ") = " << g.N(v, i) << " != ";
          exit(0);
        }
      }
    }
    assert(deg==c_deg);
    for (vidType i = 0; i < deg; i ++) {
      if (g.N(v, i) != adj_v[i]) {
        std::cout << "g.N(" << v << "," << i << ") = " << g.N(v, i) << " != ";
        std::cout << "cg.N(" << v << "," << i << ") = " << adj_v[i] << "\n";
      }
      assert(g.N(v, i)==adj_v[i]);
    }
  }
  return 0;
}
