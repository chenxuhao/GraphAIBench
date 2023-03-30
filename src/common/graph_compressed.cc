#include "graph.h"
#include "codecfactory.h"
using namespace SIMDCompressionLib;

template<bool map_vertices, bool map_edges>
vidType GraphT<map_vertices, map_edges>::decode_vertex_hybrid(vidType v, vidType* ptr, std::string scheme) {
  auto degree = read_degree(v);
  if (degree > degree_threshold) { // vbyte
    decode_vertex_vbyte(v, ptr, scheme);
  } else { // unary
    decode_vertex_unary(v, ptr, degree);
  }
  return degree;
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::decode_vertex_unary(vidType v, vidType* out, vidType degree) {
  auto start = vertices_compressed[v];
  auto in = &edges_compressed[start];
  //uint32_t count = *(uint32_t *)in;
  //in ++;
  uint64_t offset = 0;
  UnaryDecoder decoder(in, offset);
  vidType x = decoder.decode_residual_code();
  out[0] = (x & 1) ? v - (x >> 1) - 1 : v + (x >> 1);
  for (vidType i = 1; i < degree; i++) {
    out[i] = out[i-1] + decoder.decode_residual_code() + 1;
  }
}

template<bool map_vertices, bool map_edges>
vidType GraphT<map_vertices, map_edges>::decode_vertex_vbyte(vidType v, vidType* out, std::string scheme) {
  assert(v >= 0 && v < V());
  auto start = vertices_compressed[v];
  auto length = vertices_compressed[v+1] - start;
  auto in = &edges_compressed[start];
  shared_ptr<IntegerCODEC> schemeptr = CODECFactory::getFromName(scheme);
  size_t deg = 0;
  schemeptr->decodeArray(in, length, out, deg);
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

  if (scheme == "hybrid") {
  } else if (scheme == "cgr") {
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
      offset += deg;
      vertices[v+1] = offset;
    }
  }
  t.Stop();
  std::cout << "Graph decompressed time: " << t.Seconds() << "\n";
}

template<bool map_vertices, bool map_edges>
VertexSet GraphT<map_vertices, map_edges>::N_hybrid(vidType vid, std::string scheme) {
  assert(vid >= 0);
  assert(vid < V());
  VertexSet adj(vid);
  vidType deg = 0;
  deg = decode_vertex_hybrid(vid, adj.data(), scheme);
  adj.adjust_size(deg);
  return adj;
}

template<bool map_vertices, bool map_edges>
VertexSet GraphT<map_vertices, map_edges>::N_vbyte(vidType vid, std::string scheme) {
  assert(vid >= 0);
  assert(vid < V());
  VertexSet adj(vid);
  vidType deg = 0;
  deg = decode_vertex_vbyte(vid, adj.data(), scheme);
  assert(deg <= max_degree);
  adj.adjust_size(deg);
  return adj;
}

template class GraphT<false, false>;
template class GraphT<false, true>;
template class GraphT<true, true>;
