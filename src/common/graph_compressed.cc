#include "graph.h"
#include "cgr_decoder.hh"
#include "vbyte_decoder.hh"
#include <endian.h>

// Intel CPU uses little endian
#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

//#include "codecfactory.h"
//using namespace SIMDCompressionLib;

template<> void GraphT<>::load_compressed_graph(std::string prefix, std::string scheme, bool permutated) {
  // read meta information
  read_meta_info(prefix);
  assert(max_degree > 0 && max_degree < n_vertices);
  std::cout << "Reading compressed graph: |V| " << n_vertices << " |E| " << n_edges << "\n";
  VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);

  if (scheme == "hybrid") {
    degrees.resize(n_vertices);
    std::string degree_filename = prefix+".degree.bin";
    //std::cout << "Hybrid scheme: Loading degrees from file " << degree_filename << "\n";
    std::ifstream inf(degree_filename.c_str(), std::ios::binary);
    assert(inf.good());
    inf.read(reinterpret_cast<char*>(degrees.data()), sizeof(vidType) * n_vertices);
    inf.close();
    //for (int i = 0; i < 8; i++)
    //  std::cout << "Debug: degrees[" << i << "]=" << degrees[i] << "\n";
  }

  // load row offsets
  read_file(prefix+".vertex.bin", vertices_compressed, n_vertices+1);
  //std::cout << "Vertex pointers file loaded!\n";
  //for (vidType v = 0; v < n_vertices+1; v++)
  //  std::cout << "rowptr[" << v << "]=" << vertices_compressed[v] << "\n";

  // load edges, i.e., column indices
  std::ifstream ifs;
  ifs.open(prefix+".edge.bin", std::ios::in | std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    std::cout << "open graph file failed!" << std::endl;
    exit(1);
  }
  std::streamsize num_bytes = ifs.tellg();
  //std::cout << "Loading edgelists file (" << num_bytes << " bytes)\n";
  ifs.seekg(0, std::ios::beg);
  edges_compressed.clear();
  is_compressed_ = true;

  // vbyte or hybrid encoding, read binary directly
  if (scheme != "cgr" || permutated) {
    edges_compressed.resize((num_bytes-1)/vid_size+1);
    ifs.read((char*)edges_compressed.data(), num_bytes);
    ifs.close();
    return;
  }

  // cgr encoding; permutate bytes within each word
  std::cout << "This graph is not pre-permutated; permutate it now as we read it from disk (due to little-endian in Intel CPU)\n";
  auto res_bytes = num_bytes % vid_size;
  std::vector<uint8_t> buffer(num_bytes);
  ifs.read((char*) buffer.data(), num_bytes);
  auto num_words = (num_bytes-res_bytes)/vid_size + (res_bytes?1:0);
  edges_compressed.resize(num_words);
  assert(vid_size == 4);
  #pragma omp parallel for
  for (int64_t i = 0; i < num_words; i++) {
    vidType tmp = 0;
    auto idx = i << 2; // i times 4
    for (int j = 0; j < vid_size; j++) {
      if (idx+j>=num_bytes) break;
      tmp <<= 8;
      tmp += buffer[idx+j];
      if (j == vid_size-1)
        edges_compressed[i] = tmp;
    }
  }
  vidType tmp = 0;
  for (int i = 0; i < res_bytes; i++) {
    tmp <<= 8;
    tmp += buffer[num_bytes+i-res_bytes];
  }
  if (res_bytes) {
    int rem = res_bytes;
    while (rem % vid_size)
      tmp <<= 8, rem++;
    edges_compressed[num_words-1] = tmp;
  }

  ifs.close();
  //std::cout << "Edgelists file loaded!\n";
}

template<bool map_vertices, bool map_edges>
vidType GraphT<map_vertices, map_edges>::decode_vertex_hybrid(vidType v, vidType* out, std::string scheme, bool use_segment) {
  auto degree = read_degree(v);
  if (degree == 0) return 0;
  if (degree > degree_threshold) { // vbyte
    //decode_vertex_vbyte(v, out, scheme);
    auto start = vertices_compressed[v];
    auto in = &edges_compressed[start];
    vbyte_decoder decoder(scheme);
    decoder.decode(degree, in, out);
  } else { // unary
    if (use_segment)
      decode_vertex_unary_segmented(v, out, degree);
    else
      decode_vertex_unary(v, out, degree);
  }
  return degree;
}

template<bool map_vertices, bool map_edges>
vidType GraphT<map_vertices, map_edges>::decode_vertex_vbyte(vidType v, vidType* out, std::string scheme) {
  assert(v >= 0 && v < V());
  auto start = vertices_compressed[v];
  auto in = &edges_compressed[start];
  size_t deg = 0;
  vbyte_decoder decoder(scheme);
  deg = decoder.decode(in, out);
  //auto length = vertices_compressed[v+1] - start;
  //shared_ptr<IntegerCODEC> schemeptr = CODECFactory::getFromName(scheme);
  //schemeptr->decodeArray(in, length, out, deg);
  assert(deg <= max_degree);
  return vidType(deg);
}

template<bool map_vertices, bool map_edges>
vidType GraphT<map_vertices, map_edges>::decode_vertex_cgr(vidType v, vidType* out) {
  auto in = &edges_compressed[0];
  auto offset = vertices_compressed[v];
  cgr_decoder decoder(v, in, offset, out);
  return decoder.decode();
}

// not word-aligned
template<bool map_vertices, bool map_edges>
vidType GraphT<map_vertices, map_edges>::decode_vertex_unary(vidType v, vidType* out) {
  auto in = &edges_compressed[0];
  auto begin = vertices_compressed[v];
  auto end = vertices_compressed[v+1];
  //std::cout << "decoding vertex " << v << " in_ptr=" << in << " out_ptr=" << out << " begin = " << begin << " end = " << end << "\n";
  if (begin == end) return 0;
  UnaryDecoder decoder(in, begin);
  // decode the first element
  vidType x = decoder.decode_residual_code();
  out[0] = (x & 1) ? v - (x >> 1) - 1 : v + (x >> 1);
  //std::cout << "\t out[0] = " << out[0] << " ";
  // decode the rest of elements
  vidType i = 1;
  while (decoder.get_offset() < end) {
    out[i] = out[i-1] + decoder.decode_residual_code() + 1;
    //std::cout << "out[" << i << "] = " << out[i] << " ";
    i++;
  }
  //std::cout << "\n";
  return i;
}

// word-aligned
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::decode_vertex_unary(vidType v, vidType* out, vidType degree) {
  auto in = &edges_compressed[0];
  auto offset = vertices_compressed[v] * 32; // transform word-offset to bit-offset
  UnaryDecoder decoder(in, offset);
  // decode the first element
  vidType x = decoder.decode_residual_code();
  out[0] = (x & 1) ? v - (x >> 1) - 1 : v + (x >> 1);
  // decode the rest of elements
  for (vidType i = 1; i < degree; i++) {
    out[i] = out[i-1] + decoder.decode_residual_code() + 1;
  }
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::decode_vertex_unary_segmented(vidType v, vidType* out, vidType degree) {
  auto in = &edges_compressed[0];
  auto offset = vertices_compressed[v] * 32;
  //std::cout << "decoding vertex " << v << " in_ptr=" << in << " out_ptr=" << out << " degree = " << degree << "\n";
  cgr_decoder decoder(v, in, offset, out);
  auto deg = decoder.decode();
  //if (deg != degree) {
  //  std::cout << "deg = " << deg << "\n";
  //  exit(1);
  //}
  //assert(deg == degree);
}

template<bool map_vertices, bool map_edges>
VertexSet GraphT<map_vertices, map_edges>::N_hybrid(vidType vid, std::string scheme, bool use_segment) {
  assert(vid >= 0);
  assert(vid < V());
  VertexSet adj(vid);
  vidType deg = 0;
  deg = decode_vertex_hybrid(vid, adj.data(), scheme, use_segment);
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

// TODO: this method does not work for now
template<bool map_vertices, bool map_edges>
VertexSet GraphT<map_vertices, map_edges>::N_cgr(vidType vid, bool use_segment) {
  assert(vid >= 0);
  assert(vid < n_vertices);
  VertexSet adj(vid);
  vidType deg = 0;
  if (use_segment) {
    deg = decode_vertex_cgr(vid, adj.data());
  } else {
#ifdef USE_INTERVAL
    std::cout << "non-segmented interval not supported yet!\n";
    exit(1);
#endif
    deg = decode_vertex_unary(vid, adj.data());
  }
  assert(deg <= max_degree);
  adj.adjust_size(deg);
#ifdef USE_INTERVAL
  adj.sort();
#endif
  return adj;
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
  degrees.resize(n_vertices);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    VertexSet adj(v);
    degrees[v] = decode_vertex_cgr(v, adj.data());
  }
  parallel_prefix_sum<vidType,eidType>(degrees, vertices);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    auto offset = vertices[v];
    auto deg = decode_vertex_cgr(v, &edges[offset]);
    std::sort(edges+offset, edges+offset+deg);
  }
#else 
    for (vidType v = 0; v < n_vertices; v++) {
      offset += decode_vertex_cgr(v, &edges[offset]);
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

template <bool map_vertices, bool map_edges>
VertexSet GraphT<map_vertices, map_edges>::get_interval_neighbors(vidType v) {
  VertexSet adj(v);
  auto in_ptr = &edges_compressed[0];
  auto off = vertices_compressed[v];
  cgr_decoder decoder(v, in_ptr, off, adj.data());
  std::cout << "before decode_intervals: global_offset = " << decoder.get_offset() << "\n";
  vidType num_neighbors = decoder.decode_intervals();
  std::cout << "after decode_intervals: global_offset = " << decoder.get_offset() << "\n";
  assert(num_neighbors <= max_degree);
  adj.adjust_size(num_neighbors);
  return adj;
}

template <bool map_vertices, bool map_edges>
vidType GraphT<map_vertices,map_edges>::intersect_num_compressed(VertexSet& vs, vidType u) {
  vidType num = 0;
  auto offset = vertices_compressed[u];
  auto in_ptr = &edges_compressed[0];
  cgr_decoder u_decoder(u, in_ptr, offset);
  //std::cout << "intersect_num_compressed() vertex " << u << " in_ptr=" << in_ptr << "\n";

  Timer t;
  t.Start();
#ifdef USE_INTERVAL
  VertexList u_begins, u_ends;
  u_decoder.decode_intervals(u_begins, u_ends);
#endif
  VertexSet u_residuals(u);
  auto u_deg_res = u_decoder.decode_residuals(0, u_residuals.data());
  u_residuals.adjust_size(u_deg_res);
  t.Stop();
  timers[DECOMPRESS] += t.Seconds();

  t.Start();
#ifdef USE_INTERVAL
  // compare vs and u_itv
  num += intersection_num(vs, u_begins, u_ends);
#endif
  // compare vs and u_res
  num += intersection_num(vs, u_residuals);
  t.Stop();
  timers[SETOPS] += t.Seconds();
  return num;
}

template <bool map_vertices, bool map_edges>
vidType GraphT<map_vertices,map_edges>::intersect_num_compressed(VertexSet& vs, vidType u, vidType up) {
  vidType num = 0;
  auto offset = vertices_compressed[u];
  auto in_ptr = &edges_compressed[0];
  cgr_decoder u_decoder(u, in_ptr, offset);

  Timer t;
  t.Start();
#ifdef USE_INTERVAL
  VertexList u_begins, u_ends;
  u_decoder.decode_intervals(u_begins, u_ends);
#endif
  VertexSet u_residuals(u);
  auto u_deg_res = u_decoder.decode_residuals(0, u_residuals.data());
  u_residuals.adjust_size(u_deg_res);
  t.Stop();
  timers[DECOMPRESS] += t.Seconds();

  t.Start();
#ifdef USE_INTERVAL
  // compare vs and u_itv
  num += intersection_num(vs, u_begins, u_ends, up);
#endif
  // compare vs and u_res
  num += intersection_num(vs, u_residuals, up);
  t.Stop();
  timers[SETOPS] += t.Seconds();
  return num;
}

template <bool map_vertices, bool map_edges>
vidType GraphT<map_vertices,map_edges>::intersect_num_compressed(vidType v, vidType u) {
  vidType num = 0;
  auto uoff = vertices_compressed[u];
  auto voff = vertices_compressed[v];
  auto in_ptr = &edges_compressed[0];
  cgr_decoder u_decoder(u, in_ptr, uoff);
  cgr_decoder v_decoder(v, in_ptr, voff);

#ifdef USE_INTERVAL
  VertexList u_begins, u_ends;
  VertexList v_begins, v_ends;
  v_decoder.decode_intervals(v_begins, v_ends);
  u_decoder.decode_intervals(u_begins, u_ends);
#endif
  VertexSet v_residuals(v);
  VertexSet u_residuals(u);
  auto v_deg_res = v_decoder.decode_residuals(0, v_residuals.data());
  v_residuals.adjust_size(v_deg_res);
  auto u_deg_res = u_decoder.decode_residuals(0, u_residuals.data());
  u_residuals.adjust_size(u_deg_res);

#ifdef USE_INTERVAL
   // compare v_itv and u_itv
  num += intersection_num(v_begins, v_ends, u_begins, u_ends);
  // compare v_itv and u_res
  num += intersection_num(u_residuals, v_begins, v_ends);
  // compare v_res and u_itv
  num += intersection_num(v_residuals, u_begins, u_ends);
#endif
  // compare v_res and u_res
  num += intersection_num(v_residuals, u_residuals);
  return num;
}
 
template <bool map_vertices, bool map_edges>
vidType GraphT<map_vertices,map_edges>::intersect_num_compressed(vidType v, vidType u, vidType up) {
  vidType num = 0;
  auto uoff = vertices_compressed[u];
  auto voff = vertices_compressed[v];
  auto in_ptr = &edges_compressed[0];
  cgr_decoder u_decoder(u, in_ptr, uoff);
  cgr_decoder v_decoder(v, in_ptr, voff);

#ifdef USE_INTERVAL
  VertexList v_begins, v_ends;
  VertexList u_begins, u_ends;
  v_decoder.decode_intervals(v_begins, v_ends);
  u_decoder.decode_intervals(u_begins, u_ends);
#endif
  VertexSet v_residuals(v);
  VertexSet u_residuals(u);
  auto v_deg_res = v_decoder.decode_residuals(0, v_residuals.data());
  v_residuals.adjust_size(v_deg_res);
  auto u_deg_res = u_decoder.decode_residuals(0, u_residuals.data());
  u_residuals.adjust_size(u_deg_res);

#ifdef USE_INTERVAL
   // compare v_itv and u_itv
  num += intersection_num(v_begins, v_ends, u_begins, u_ends, up);
  // compare v_itv and u_res
  num += intersection_num(u_residuals, v_begins, v_ends, up);
  // compare v_res and u_itv
  num += intersection_num(v_residuals, u_begins, u_ends, up);
#endif
  // compare v_res and u_res
  num += intersection_num(v_residuals, u_residuals, up);
  return num;
}

template<> void GraphT<>::print_compressed_colidx() {
  // print compressed colidx
  for (vidType v = 0; v < n_vertices; v++) {
    auto begin = vertices_compressed[v];
    auto end = vertices_compressed[v+1];
    auto bit_len = end-begin;
    std::cout << "vertex " << v << " neighbor list (" << bit_len << " bits): ";
    vidType offset = begin % 32;
    auto first_word = edges_compressed[begin / 32];
    int num_words = 1;
    if (bit_len > (32-offset))
      num_words += (bit_len - (32-offset) -1)/32 + 1;
    first_word <<= offset;
    first_word >>= offset;
    std::bitset<32> x(first_word);
    std::cout << x << "<" << std::min(bit_len, eidType(32-offset)) << "> ";
    int i = 1;
    for (; i < num_words-1; i++) {
      std::bitset<32> y(edges_compressed[i+begin/32]);
      std::cout << y << "<32> ";
    }
    if (num_words > 1) {
      auto last_word = edges_compressed[i+begin/32];
      if (end%32) {
        offset = 32 - (end % 32);
        last_word >>= offset;
      }
      std::bitset<32> y(last_word);
      std::cout << y << "<" << ((end%32)?(end%32):32) << "> ";
    }
    std::cout << "\n";
  }
}

template class GraphT<false, false>;
template class GraphT<false, true>;
template class GraphT<true, true>;
