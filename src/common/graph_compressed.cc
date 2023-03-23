#include "graph.h"
#include "codecfactory.h"
using namespace SIMDCompressionLib;
template<bool map_vertices, bool map_edges>
vidType GraphT<map_vertices, map_edges>::decode_vertex_vbyte(vidType v, vidType* ptr, std::string scheme) {
  assert(v >= 0 && v < V());
  auto start = vertices_compressed[v];
  auto length = vertices_compressed[v+1] - start;
  shared_ptr<IntegerCODEC> schemeptr = CODECFactory::getFromName(scheme);
  size_t deg = 0;
  schemeptr->decodeArray(&edges_compressed[start], length, ptr, deg);
  assert(deg <= max_degree);
  return vidType(deg);
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::decompress(std::string scheme) {
  std::cout << "Decompressing the graph (format=" << scheme << ")\n";
  Timer t;
  t.Start();
  vertices = new eidType[n_vertices+1];
  edges = new vidType[n_edges];
  vertices[0] = 0;
  eidType offset = 0;

  if (scheme == "cgr") {
#if 0
  VertexList degrees(n_vertices);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    VertexSet adj(v);
    degrees[v] = decode_vertex(v, adj.data());
  }
  parallel_prefix_sum<vidType,eidType>(degrees, vertices);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    auto offset = vertices[v];
    auto deg = decode_vertex(v, &edges[offset]);
    std::sort(edges+offset, edges+offset+deg);
  }
#else 
    for (vidType v = 0; v < n_vertices; v++) {
      offset += decode_vertex(v, &edges[offset]);
      vertices[v+1] = offset;
      std::sort(edges+vertices[v], edges+offset);
    }
#endif
  } else {
    // VByte format
    for (vidType v = 0; v < n_vertices; v++) {
      auto deg = decode_vertex_vbyte(v, &edges[offset], scheme);
      //std::cout << "decoding vertex " << v << " degree=" << deg << "\n";
      offset += deg;
      vertices[v+1] = offset;
    }
  }
  t.Stop();
  std::cout << "Graph decompressed time: " << t.Seconds() << "\n";
}

template<bool map_vertices, bool map_edges>
VertexSet GraphT<map_vertices, map_edges>::N_vbyte(vidType vid, std::string scheme) {
  assert(vid >= 0);
  assert(vid < V());
  VertexSet adj(vid);
  auto deg = decode_vertex_vbyte(vid, adj.data(), scheme);
  assert(deg <= max_degree);
  adj.adjust_size(deg);
  return adj;
}

template class GraphT<false, false>;
template class GraphT<false, true>;
template class GraphT<true, true>;
