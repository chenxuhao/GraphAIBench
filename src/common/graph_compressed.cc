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

