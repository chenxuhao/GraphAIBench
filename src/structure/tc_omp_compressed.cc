#include "graph.h"
#include <omp.h>

void TCSolver(Graph &g, uint64_t &total, std::string scheme, bool segmented);

void printusage(std::string bin) {
  std::cout << "Try " << bin << " -s name-of-scheme(cgr) -i ../../inputs/mico/dag-streamvbyte [-o (oriented)] [-p (permutated)]\n";
}

int main(int argc,char *argv[]) {
  std::string schemename = "cgr";
  std::string filename = "";
  bool permutated = false;
  bool oriented = false;
  bool segmented = false;
  vidType degree_threshold = 32;
  int c;
  while ((c = getopt(argc, argv, "s:i:opgd:h")) != -1) {
    switch (c) {
      case 's':
        schemename = optarg;
        break;
      case 'i':
        filename = optarg;
        break;
      case 'o':
        oriented = true;
        break;
      case 'p':
        permutated = true;
        break;
      case 'g':
        segmented = true;
        break;
      case 'd':
        degree_threshold = atoi(optarg);
        break;
      case 'h':
        printusage(argv[0]);
        return 0;
      default:
        abort();
    }
  }
  if (argc < 3) {
    std::cout << "# arguments (" << argc << ") incorrect\n";
    printusage(argv[0]);
    return -1;
  }
  if (!oriented) {
    std::cout << "Graph must be oriented\n";
    printusage(argv[0]);
    return -1;
  }
 
  Graph g;
  g.set_degree_threshold(degree_threshold);
  if (schemename == "decomp") // uncompressed graph
    g.load_graph(filename);
  else
    g.load_compressed_graph(filename, schemename, permutated);
  g.print_meta_data();

  uint64_t total = 0;
  TCSolver(g, total, schemename, segmented);
  std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}

void TCSolver(Graph &g, uint64_t &total, std::string scheme, bool segmented) {
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

  if (scheme == "decomp") { // decompressed graph
    #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
    for (vidType u = 0; u < g.V(); u ++) {
      auto adj_u = g.N(u);
      for (auto v : adj_u) {
        auto adj_v = g.N(v);
       auto num = (uint64_t)intersection_num(adj_u, adj_v);
        counter += num;
      }
    }
  } else if (scheme == "cgr") { // cgr graph
    #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
    for (vidType u = 0; u < g.V(); u ++) {
      auto adj_u = g.N_cgr(u, segmented);
      for (auto v : adj_u) {
        auto adj_v = g.N_cgr(v, segmented);
        auto num = (uint64_t)intersection_num(adj_u, adj_v);
        counter += num;
      }
    }
  } else if (scheme == "hybrid") { // hybrid scheme: unary + vbyte
    std::string vbyte_scheme = "streamvbyte";
    #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
    for (vidType u = 0; u < g.V(); u ++) {
      auto adj_u = g.N_hybrid(u, vbyte_scheme, segmented);
      for (auto v : adj_u) {
        auto adj_v = g.N_hybrid(v, vbyte_scheme, segmented);
        auto num = (uint64_t)intersection_num(adj_u, adj_v);
        counter += num;
      }
    }
  } else { // vbyte graph
    #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
    for (vidType u = 0; u < g.V(); u ++) {
      auto adj_u = g.N_vbyte(u, scheme);
      for (auto v : adj_u) {
        auto adj_v = g.N_vbyte(v, scheme);
       auto num = (uint64_t)intersection_num(adj_u, adj_v);
        counter += num;
      }
    }
  }
  t.Stop();
  std::cout << "runtime [tc_omp_" << scheme << "] = " << t.Seconds() << " sec\n";
  //std::cout << "total_num_triangles = " << counter << "\n";
  //std::cout << "Set operations time: "   << timers[SETOPS] << "\n";
  //std::cout << "Decompress time: "   << timers[DECOMPRESS] << "\n";
  total = counter;
  return;
}

