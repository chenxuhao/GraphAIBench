#include "graph.h"
#include "codecfactory.h"

using namespace SIMDCompressionLib;

VertexSet N_vbyte(Graph &g, vidType vid) {
  assert(vid >= 0);
  assert(vid < g.V());
  vidType max_deg = g.get_max_degree();
  VertexSet adj(vid);
  auto vertices_compressed = g.rowptr_compressed();
  auto edges_compressed = g.colidx_compressed();
  auto start = vertices_compressed[vid];
  auto length = vertices_compressed[vid+1] - start;
  std::string schemename = "streamvbyte";
  size_t osize(max_deg);
  shared_ptr<IntegerCODEC> schemeptr = CODECFactory::getFromName(schemename);
  //std::cout << "decoding vertex " << vid << " 's neighbor list, starting at " << start << " with " << length << " bytes\n";
  //std::cout << "debugging: edges_compressed " << edges_compressed << " adj.data() " << adj.data() << "\n";
  schemeptr->decodeArray(&edges_compressed[start], length, adj.data(), osize);
  assert(osize <= max_deg);
  adj.adjust_size(osize);
  return adj;
}

void TCSolver(Graph &g, uint64_t &total, bool compressed) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Triangle Counting on Compressed Graph (" << num_threads << " threads)\n";
  uint64_t counter = 0;
  //timers[SETOPS] = 0.0;
  //timers[DECOMPRESS] = 0.0;
  Timer t;
  t.Start();
  if (compressed) {
    #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
    for (vidType u = 0; u < g.V(); u ++) {
      auto adj_u = N_vbyte(g, u);
      for (auto v : adj_u) {
        auto adj_v = N_vbyte(g, v);
        auto num = (uint64_t)intersection_num(adj_u, adj_v);
        counter += num;
      }
    }
  } else {
    #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
    for (vidType u = 0; u < g.V(); u ++) {
      auto adj_u = g.N(u);
      for (auto v : adj_u) {
        auto adj_v = g.N(v);
        auto num = (uint64_t)intersection_num(adj_u, adj_v);
        counter += num;
      }
    }
  }
  t.Stop();
  std::cout << "runtime [tc_omp_vbyte] = " << t.Seconds() << " sec\n";
  std::cout << "total_num_triangles = " << counter << "\n";
  //std::cout << "Set operations time: "   << timers[SETOPS] << "\n";
  //std::cout << "Decompress time: "   << timers[DECOMPRESS] << "\n";
  total = counter;
  return;
}

int main(int argc,char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> \n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph\n";
    abort();
  }
  Graph g;
  g.load_compressed_graph(argv[1], false);
  g.print_meta_data();

  uint64_t total = 0;
  //g.decompress("streamvbyte");
  //TCSolver(g, total, false);
  //std::cout << "total_num_triangles = " << total << "\n";

  total = 0;
  TCSolver(g, total, true);
  std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}

