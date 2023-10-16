#include "graph.h"
// writing on a text file
#include <iostream>
#include <fstream>
#include "khop.h"
using namespace std;

inline std::vector<uint32_t> get_initial_sample(uint64_t seeds_size, uint64_t graph_size);
inline std::unordered_map<uint32_t, uint32_t> get_initial_sample_adj_hash_map(const std::vector<uint32_t>& n_ids);
inline std::tuple<std::vector<vidType, std::allocator<vidType>>, std::vector<vidType, std::allocator<vidType>>, std::vector<vidType, std::allocator<vidType>>> sample_adj(vector<eidType> rowptr, vector<vidType> col, std::vector<uint32_t> n_ids, std::vector<uint32_t>& all_ids, std::unordered_map<uint32_t, uint32_t>& n_id_map, uint32_t num_neighbors, bool replace);

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <graph>"
              << "[num_gpu(1)] [chunk_size(1024)]\n";
    cout << "Example: " << argv[0] << " ../inputs/cora/graph\n";
    exit(1);
  }

  Graph g(argv[1], 0, 0, 0, 0, 0);
  eidType* rptrs = g.rowptr();
  vector<eidType> row_ptrs(rptrs, rptrs + g.V());
  row_ptrs.push_back(g.E());
  vidType* cptrs = g.colidx();
  vector<vidType> col_idxs(cptrs, cptrs + g.E());
  vector<uint32_t> sample_per_round = {2, 3};
  int sample_rounds = sample_per_round.size();
  uint64_t initial_size = 2;
  vector<uint32_t> n_ids = get_initial_sample(initial_size, g.V());
  vector<uint32_t> all_ids = n_ids;
  unordered_map<uint32_t, uint32_t> n_id_map = get_initial_sample_adj_hash_map(n_ids);
  vector<pair<uint32_t, uint32_t>> edges;
  for(int round = 0; round < sample_rounds; round++) {
    auto sampled = sample_adj(row_ptrs, col_idxs, n_ids, all_ids, n_id_map, sample_per_round[round], true);
    vector<uint32_t> out_rowptrs = get<0>(sampled);
    vector<uint32_t> out_cols = get<1>(sampled);
    all_ids = get<2>(sampled);
    // for (int i = 0; i < out_rowptrs.size(); i++) {
    //   cout << "out_rowptrs: " << out_rowptrs[i] << endl;
    // }
    //     for (int i = 0; i < out_cols.size(); i++) {
    //   cout << "out_cols: " << out_cols[i] << endl;
    // }
    // save edges
    for (int i = 0; i < out_cols.size(); i++) {
      uint32_t r = i/sample_per_round[round];
      uint32_t src_id = n_ids[r];
      // store edges in terms of new subgraph node ids
      uint32_t new_src_id = n_id_map[src_id];
      uint32_t new_dst_id = n_id_map[out_cols[i]];
      // pair<uint32_t, uint32_t> curr_edge (n_id_map[src_id], n_id_map[out_cols[i]]);
      // cout << "curr_edge: " << curr_edge.first << ", " << curr_edge.second << endl;
      edges.push_back({new_src_id, new_dst_id});
    }
    n_ids = out_cols;
  }
  cout << "New ids to old ids mapping" << endl;
  for (auto id: n_id_map) {
    cout << id.first << ": " << id.second << endl;
  }
  cout << "SAMPLING DONE..." << endl;
  int nv = all_ids.size();
  int ne = edges.size();
  sort(edges.begin(), edges.end());
  cout << "CHECKING EDGES" << endl;
  for (auto edge: edges) {
    cout << edge.first << " " << edge.second << endl;
  }
  vector<uint32_t> degrees(nv, 0);
  cout << "Getting degrees..." << endl;
  cout << "degrees size: " << nv << endl;
  for (auto edge: edges) {
    cout << "src node: " << edge.first << endl;
    degrees[edge.first]++;
  }
  cout << "Getting offsets..." << endl;
  auto offsets = utils::prefix_sum(degrees);

  cout << "BUILDING NEW SUBGRAPH..." << endl;
  Graph subgraph;
  subgraph.allocateFrom(nv, ne);
  for (size_t src = 0; src < nv; src++) {
    subgraph.fixEndEdge(src, offsets[src+1]);
    int idx;
    for (idx = offsets[src]; idx < offsets[src+1]; idx++) {
      subgraph.constructEdge(idx, edges[idx].second);
    }
  }
  cout << "nv: " << subgraph.num_vertices() << ", ne: " << subgraph.num_edges() << endl;

  ofstream myfile ("../src/sampling/subgraph.txt");
  // write out all edges to file
  if (myfile.is_open())
  cout << "Saving subgraph" << endl;
  {
    for (auto edge: edges) {
      myfile << edge.first << "-" << edge.second << "\n";
    }
    myfile.close();
  }

  return 0;
};