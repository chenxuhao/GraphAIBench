// Copyright 2020, MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph_partition.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> [num_gpu(1)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "Test graph partitioning.\n";
  int n_devices = 2;
  if (argc > 2) n_devices = atoi(argv[2]);
  Graph g(argv[1], 1);
  g.print_meta_data();
  auto nv = g.V();

  // partition the graph
  PartitionedGraph pg(&g, n_devices);
#ifdef USE_INDUCED
  pg.edgecut_induced_partition1D();
#else
  pg.edgecut_partition1D();
#endif
  //pg.print_subgraphs();

  Timer t;
  t.Start();
  uint64_t counter = 0;
#ifdef USE_INDUCED
  for (int i = 0; i < pg.get_num_subgraphs(); i++) {
    auto sg = pg.get_subgraph(i);
    auto begin = pg.get_local_begin(i);
    auto end = pg.get_local_end(i);
    std::cout << "subgraph[" << i << "]: from vertex " << begin << " to " << end << "\n";
    #pragma omp parallel for reduction(+ : counter) schedule(dynamic, 1)
    for (vidType u = begin; u < end; u ++) {
      auto adj_u = sg->N(u);
      for (auto v : adj_u) {
        counter += (uint64_t)intersection_num(adj_u, sg->N(v));
      }
    }
  }
#else
  for (int i = 0; i < pg.get_num_subgraphs(); i++) {
    auto sg = pg.get_subgraph(i);
    auto num_subgraphs = pg.get_num_subgraphs();
    int subgraph_size = (nv-1) / num_subgraphs + 1;
    vidType begin_vid = i * subgraph_size;
    vidType end_vid = (i+1) * subgraph_size;
    auto offset = sg->edge_begin(i * subgraph_size);
    for (vidType u = begin_vid; u < end_vid; u ++) {
      //auto adj_u = sg->N(u);
      auto ua = sg->edge_begin(u);
      auto ub = sg->edge_end(u);
      VertexSet adj_u;
      for (int j = ua-offset; j <ub-offset; j++) {
        adj_u.add(sg->getEdgeDst(j));
      }
      for (auto v : adj_u) {
        auto sg_id = v / subgraph_size;
        auto dst_sg = pg.get_subgraph(sg_id);
        auto va = dst_sg->edge_begin(v);
        auto vb = dst_sg->edge_end(v);
        auto first_vertex = sg_id * subgraph_size;
        auto dst_offset = dst_sg->edge_begin(first_vertex);
        VertexSet adj_v;
        for (int j = va-dst_offset; j < vb-dst_offset; j++) {
          adj_v.add(dst_sg->getEdgeDst(j));
        }
        counter += (uint64_t)intersection_num(adj_u, adj_v);
      }
    }
  }
#endif
  t.Stop();
  std::cout << "total_num_triangles = " << counter << "\n";
  std::cout << "runtime [tc_omp_base] = " << t.Seconds() << " sec\n";
  return 0;
}

