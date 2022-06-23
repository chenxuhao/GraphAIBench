#include "utils.h"
#include "sampler.h"

//! debug function: prints out sets of vertices
void print_vertex_set(VertexSet vertex_set) {
  unsigned counter = 0;
  unsigned n       = vertex_set.size();
  std::cout << "( ";
  for (int i : vertex_set) {
    counter++;
    if (counter > 16 && counter < n - 16)
      continue;
    std::cout << i << " ";
  }
  std::cout << ")\n";
}
/*
size_t Sampler::selectVertices(index_t n, VertexSet& vertex_set, unsigned seed) {
  auto nv = full_graph->size();
  VertexList vertices(nv);
  #pragma omp parallel for
  for (index_t i = 0; i < nv; i++)
    vertices[i] = i;
  srand(seed);
  return selectVertices(nv, n, full_graph, vertices, vertex_set); 
}
*/

// API function for user-defined selection strategy
// Select n vertices from `vertices` and put them in `vertex_set`.
// nv: number of vertices in the original graph;
// n: number of vertices in the subgraph;
// m: number of vertices in the frontier.
size_t Sampler::selectVertices(index_t nv, index_t n, Graph* g, VertexList vertices, VertexSet& vertex_set) {
  // "Select a vertex set of size ", n, " from ", nv, " vertices, graph size: ", g->size(), "\n");
  assert(nv == vertices.size());
  // randomly select m vertices from vertices as frontier
  auto frontier_indices = utils::select_k_items((int)m, 0, (int)nv);
  VertexList frontier(m);
  for (index_t i = 0; i < m; i++)
    frontier[i] = vertices[frontier_indices[i]];
  vertex_set.insert(frontier.begin(), frontier.end());
  int* degrees = new int[m];
  for (index_t i = 0; i < m; i++)
    degrees[i] = (int)getDegree(g, frontier[i]);
  for (index_t i = 0; i < n - m; i++) {
    auto pos    = utils::select_one_item((int)m, degrees);
    auto u      = frontier[pos];
    auto degree = degrees[pos];
    int j       = 0;
    for (; j < degree; j++) {
      auto neighbor_id = rand() % degree; // randomly select a neighbor
      auto dst         = g->getEdgeDstHost(g->edge_begin_host(u) + neighbor_id);
      if (vertex_set.find(dst) == vertex_set.end()) {
        frontier[pos] = dst;
        degrees[pos]  = getDegree(g, frontier[pos]);
        vertex_set.insert(dst);
        break;
      }
    }
    if (j == degree) std::cout << "Not found from " << degree << " neighbors\n";
  }
  // "Done selection, vertex_set size: ", vertex_set.size(), ", set: ");
  //print_vertex_set(vertex_set);
  return vertex_set.size();
}

// Given a subset of vertices and a graph g, generate a subgraph sg from the graph g
void Sampler::reindexSubgraph(VertexSet& keptVertices, Graph& origGraph, Graph& reindexGraph) {
  auto nv = keptVertices.size(); // new graph (subgraph) size
  VertexList new_ids = reindexVertices(full_graph->size(), keptVertices);
  std::vector<uint32_t> degrees(nv, 0); // degrees of vertices in the subgraph
  for (auto v : keptVertices)
    degrees[new_ids[v]] = getDegree(&origGraph, v);
  // auto offsets = parallel_prefix_sum(degrees);
  auto offsets = utils::prefix_sum(degrees);
  auto ne      = offsets[nv];
  //std::cout << "reindex subgraph |V| = " << nv << " |E| = " << ne << "\n";
  reindexGraph.allocateFrom(nv, ne);
  //reindexGraph.constructNodes();
  VertexList old_ids(keptVertices.begin(), keptVertices.end()); // vertex ID mapping
  //#pragma omp parallel for
  for (size_t i = 0; i < nv; i++) {
    reindexGraph.fixEndEdge(i, offsets[i + 1]);
    unsigned j  = 0;
    auto old_id = old_ids[i];
    for (auto e = origGraph.edge_begin_host(old_id);
        e != origGraph.edge_end_host(old_id); e++) {
      auto dst = new_ids[origGraph.getEdgeDstHost(e)];
      assert(dst < nv);
      reindexGraph.constructEdge(offsets[i] + j, dst);
      j++;
    }
  }
}

template <typename GraphTy>
void Sampler::getMaskedDegrees(size_t n, mask_t* masks, GraphTy* g, std::vector<uint32_t>& degrees) {
  assert(degrees.size() == n);
  #pragma omp parallel for
  for (size_t src = 0; src < n; src++) {
    if (masks[src] == 1) {
      for (auto e = g->edge_begin_host(src); e != g->edge_end_host(src); e++) {
        const auto dst = g->getEdgeDstHost(e);
        if (masks[dst] == 1) degrees[src]++;
      }
    }
  }
}

// trim edges from g to form sub
template <typename GraphTy, typename SubgraphTy>
void Sampler::getMaskedGraph(index_t n, mask_t* masks, GraphTy* g, SubgraphTy* sub) {
  std::vector<uint32_t> degrees(n, 0);
  getMaskedDegrees(n, masks, g, degrees);
  // auto offsets = parallel_prefix_sum(degrees);
  auto offsets = utils::prefix_sum(degrees);
  size_t ne    = offsets[n];
  //std::cout << "subgraph |V| = " << n << " |E| = " << ne << "\n";
  sub->allocateFrom(n, ne);
  //sub->constructNodes();
  //#pragma omp parallel for
  for (size_t src = 0; src < n; src++) {
    sub->fixEndEdge(src, offsets[src + 1]);
    if (masks[src] == 1) {
      auto idx = offsets[src];
      for (auto e = g->edge_begin_host(src); e != g->edge_end_host(src); e++) {
        auto dst = g->getEdgeDstHost(e);
        if (masks[dst] == 1)
          sub->constructEdge(idx++, dst);
      }
    }
  }
}

// do the sampling of vertices from training set + using masked graph
void Sampler::generateSubgraph(VertexSet& sampledSet, mask_t* masks, Graph* sg) {
  createMasks(full_graph->size(), sampledSet, masks); // create the masks
  //for (int i = 0; i < 100; i++) std::cout << "masks[" << i << "]=" << unsigned(masks[i]) << "\n";
  Graph maskedSG; // sampled vertex set induced subgraph
  // remove edges whose destination is not masked
  getMaskedGraph(full_graph->size(), masks, full_graph, &maskedSG);
  reindexSubgraph(sampledSet, maskedSG, *sg);
  //std::cout << "subg |E| " << sg->sizeEdges() << "\n";
}

typedef int db_t;
void checkGSDB(std::vector<db_t>& DB0, std::vector<db_t>& DB1,
               std::vector<db_t>& DB2, index_t size) {
  if (DB0.capacity() < size) {
    DB0.reserve(DB0.capacity() * 2);
    DB1.reserve(DB1.capacity() * 2);
    DB2.reserve(DB2.capacity() * 2);
  }
  DB0.resize(size);
  DB1.resize(size);
  DB2.resize(size);
}

// implementation from GraphSAINT
// https://github.com/GraphSAINT/GraphSAINT/blob/master/ipdps19_cpp/sample.cpp
// n: subgraph_size; m: size_frontier
size_t Sampler::select_vertices(index_t n, VertexSet& st, unsigned seed) {
  if (n < m) m = n;
  unsigned myseed = seed;
  // DBx: Dashboard line x, IAx: Index array line x
  std::vector<db_t> DB0, DB1, DB2, IA0, IA1, IA2, IA3, IA4, nDB0, nDB1, nDB2;
  DB0.reserve(subg_deg * m * ETA);
  DB1.reserve(subg_deg * m * ETA);
  DB2.reserve(subg_deg * m * ETA);
  IA0.reserve(n);
  IA1.reserve(n);
  IA2.reserve(n);
  IA3.reserve(n);
  IA4.reserve(n);
  IA0.resize(m);
  IA1.resize(m);
  IA2.resize(m);
  IA3.resize(m);

  for (index_t i = 0; i < m; i++) {
    auto rand_idx = rand_r(&myseed) % trainingNodes.size();
    db_t v = IA3[i] = trainingNodes[rand_idx];
    st.insert(v);
    IA0[i] = getDegree(masked_graph, v);
    IA0[i] = (IA0[i] > SAMPLE_CLIP) ? SAMPLE_CLIP : IA0[i];
    IA1[i] = 1;
    IA2[i] = 0;
  }
  // calculate prefix sum for IA0 and store in IA2 to compute the address for
  // each frontier in DB
  IA2[0] = IA0[0];
  for (index_t i = 1; i < m; i++)
    IA2[i] = IA2[i - 1] + IA0[i];
  // now fill DB accordingly
  checkGSDB(DB0, DB1, DB2, IA2[m - 1]);
  for (index_t i = 0; i < m; i++) {
    db_t DB_start = (i == 0) ? 0 : IA2[i - 1];
    db_t DB_end   = IA2[i];
    for (auto j = DB_start; j < DB_end; j++) {
      DB0[j] = IA3[i];
      DB1[j] = (j == DB_start) ? (j - DB_end) : (j - DB_start);
      DB2[j] = i + 1;
    }
  }

  db_t choose, neigh_v, newsize, tmp;
  for (index_t itr = 0; itr < n - m; itr++) {
    choose = db_t(-1);
    while (choose == db_t(-1)) {
      tmp = rand_r(&myseed) % DB0.size();
      if (size_t(tmp) < DB0.size())
        if (DB0[tmp] != db_t(-1))
          choose = tmp;
    }
    choose      = (DB1[choose] < 0) ? choose : (choose - DB1[choose]);
    db_t v      = DB0[choose];
    auto degree = getDegree(masked_graph, v);
    neigh_v     = (degree != 0) ? rand_r(&myseed) % degree : db_t(-1);
    if (neigh_v != db_t(-1)) {
      neigh_v = masked_graph->getEdgeDstHost(masked_graph->edge_begin_host(v) + neigh_v);
      st.insert(neigh_v);
      IA1[DB2[choose] - 1] = 0;
      IA0[DB2[choose] - 1] = 0;
      for (auto i = choose; i < choose - DB1[choose]; i++)
        DB0[i] = db_t(-1);
      newsize = getDegree(masked_graph, neigh_v);
      newsize = (newsize > SAMPLE_CLIP) ? SAMPLE_CLIP : newsize;
    } else
      newsize = 0;
    // shrink DB to remove sampled nodes, also shrink IA accordingly
    bool cond = DB0.size() + newsize > DB0.capacity();
    if (cond) {
      // compute prefix sum for the location in shrinked DB
      IA4.resize(IA0.size());
      IA4[0] = IA0[0];
      for (size_t i = 1; i < IA0.size(); i++)
        IA4[i] = IA4[i - 1] + IA0[i];
      nDB0.resize(IA4.back());
      nDB1.resize(IA4.back());
      nDB2.resize(IA4.back());
      IA2.assign(IA4.begin(), IA4.end());
      for (size_t i = 0; i < IA0.size(); i++) {
        if (IA1[i] == 0)
          continue;
        db_t DB_start = (i == 0) ? 0 : IA4[i - 1];
        db_t DB_end   = IA4[i];
        for (auto j = DB_start; j < DB_end; j++) {
          nDB0[j] = IA3[i];
          nDB1[j] = (j == DB_start) ? (j - DB_end) : (j - DB_start);
          nDB2[j] = i + 1;
        }
      }
      // remap the index in DB2 by compute prefix of IA1 (new idx in IA)
      IA4.resize(IA1.size());
      IA4[0] = IA1[0];
      for (size_t i = 1; i < IA1.size(); i++)
        IA4[i] = IA4[i - 1] + IA1[i];
      DB0.assign(nDB0.begin(), nDB0.end());
      DB1.assign(nDB1.begin(), nDB1.end());
      DB2.assign(nDB2.begin(), nDB2.end());
      for (auto i = DB2.begin(); i < DB2.end(); i++)
        *i = IA4[*i - 1];
      db_t curr = 0;
      for (size_t i = 0; i < IA0.size(); i++) {
        if (IA0[i] != 0) {
          IA0[curr] = IA0[i];
          IA1[curr] = IA1[i];
          IA2[curr] = IA2[i];
          IA3[curr] = IA3[i];
          curr++;
        }
      }
      IA0.resize(curr);
      IA1.resize(curr);
      IA2.resize(curr);
      IA3.resize(curr);
    }
    checkGSDB(DB0, DB1, DB2, newsize + DB0.size());
    IA0.push_back(newsize);
    IA1.push_back(1);
    IA2.push_back(IA2.back() + IA0.back());
    IA3.push_back(neigh_v);
    db_t DB_start = (*(IA2.end() - 2));
    db_t DB_end   = IA2.back();
    for (auto j = DB_start; j < DB_end; j++) {
      DB0[j] = IA3.back();
      DB1[j] = (j == DB_start) ? (j - DB_end) : (j - DB_start);
      DB2[j] = IA3.size();
    }
  }
  return st.size();
}

