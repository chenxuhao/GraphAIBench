#include "graph.h"
#include "scan.h"
std::map<OPS,double> timers;

template<bool map_vertices, bool map_edges>
GraphT<map_vertices, map_edges>::GraphT(std::string prefix, bool use_dag, bool directed, 
             bool use_vlabel, bool use_elabel, bool need_reverse, bool bipartite) :
    is_directed_(directed), is_bipartite_(bipartite), is_compressed_(false), 
    max_degree(0), n_vertices(0), n_edges(0), 
    nnz(0), max_label_frequency_(0), max_label(0),
    feat_len(0), num_vertex_classes(0), num_edge_classes(0), 
    edges(NULL), vertices(NULL), vlabels(NULL), elabels(NULL), 
    features(NULL), src_list(NULL), dst_list(NULL) {
  // parse file name
  size_t i = prefix.rfind('/', prefix.length());
  if (i != string::npos) inputfile_path = prefix.substr(0, i);
  i = inputfile_path.rfind('/', inputfile_path.length());
  if (i != string::npos) name_ = inputfile_path.substr(i+1);
  std::cout << "input file path: " << inputfile_path << ", graph name: " << name_ << "\n";
  VertexSet::release_buffers();

  // read meta information
  read_meta_info(prefix, bipartite);
  // read row pointers
  if constexpr (map_vertices)
    map_file(prefix + ".vertex.bin", vertices, n_vertices+1);
  else read_file(prefix + ".vertex.bin", vertices, n_vertices+1);
  std::cout << "vertex loaded\n";
  // read column indices
  if constexpr (map_edges)
    map_file(prefix + ".edge.bin", edges, n_edges);
  else read_file(prefix + ".edge.bin", edges, n_edges);
  std::cout << "edge loaded\n";

  if (is_directed_) {
    std::cout << "This is a directed graph\n";
    if (need_reverse) {
      build_reverse_graph();
      std::cout << "This graph maintains both incomming and outgoing edge-list\n";
      has_reverse = true;
    }
  } else {
    has_reverse = true;
    reverse_vertices = vertices;
    reverse_edges = edges;
  }

  // compute maximum degree
  if (max_degree == 0) compute_max_degree();
  //else std::cout << "max_degree: " << max_degree << "\n";
  assert(max_degree > 0 && max_degree < n_vertices);

  // read vertex labels
  if (use_vlabel) {
    assert (num_vertex_classes > 0);
    assert (num_vertex_classes < 255); // we use 8-bit vertex label dtype
    std::string vlabel_filename = prefix + ".vlabel.bin";
    ifstream f_vlabel(vlabel_filename.c_str());
    if (f_vlabel.good()) {
      if constexpr (map_vertices)
        map_file(vlabel_filename, vlabels, n_vertices);
      else read_file(vlabel_filename, vlabels, n_vertices);
      std::set<vlabel_t> labels;
      for (vidType v = 0; v < n_vertices; v++)
        labels.insert(vlabels[v]);
      //for (int i = 0; i < n_vertices; i++) std::cout << unsigned(vlabels[i]) << "\n";
      std::cout << "# distinct vertex labels: " << labels.size() << "\n";
      assert(size_t(num_vertex_classes) == labels.size());
    } else {
      std::cout << "WARNING: vertex label file not exist; generating random labels\n";
      vlabels = new vlabel_t[n_vertices];
      for (vidType v = 0; v < n_vertices; v++) {
        vlabels[v] = rand() % num_vertex_classes + 1;
      }
    }
    auto max_vlabel = unsigned(*(std::max_element(vlabels, vlabels+n_vertices)));
    std::cout << "maximum vertex label: " << max_vlabel << "\n";
  }
  if (use_elabel) {
    std::string elabel_filename = prefix + ".elabel.bin";
    ifstream f_elabel(elabel_filename.c_str());
    if (f_elabel.good()) {
      assert (num_edge_classes > 0);
      if constexpr (map_edges)
        map_file(elabel_filename, elabels, n_edges);
      else read_file(elabel_filename, elabels, n_edges);
      std::set<elabel_t> labels;
      for (eidType e = 0; e < n_edges; e++)
        labels.insert(elabels[e]);
      //for (int i = 0; i < n_edges; i++) {
      //  if (elabels[i] > 5 || elabels[i] < 1)
      //    std::cout << "elabels[" << i << "]=" << elabels[i] << "\n";
      //}
      //for (int i = 0; i < 10; i++) std::cout << elabels[i] << "\n";
      std::cout << "# distinct edge labels: " << labels.size() << "\n";
      //for (auto l : labels) std::cout << l << "\n";
      assert(size_t(num_edge_classes) >= labels.size());
    } else {
      std::cout << "WARNING: edge label file not exist; generating random labels\n";
      elabels = new elabel_t[n_edges];
      if (num_edge_classes < 1) {
        num_edge_classes = 1;
        for (eidType e = 0; e < n_edges; e++) {
          elabels[e] = 1;
        }
      } else {
        for (eidType e = 0; e < n_edges; e++) {
          elabels[e] = rand() % num_edge_classes + 1;
        }
      }
    }
    auto max_elabel = unsigned(*(std::max_element(elabels, elabels+n_edges)));
    std::cout << "maximum edge label: " << max_elabel << "\n";
  }
  // orientation: convert the undirected graph into directed. Only for k-cliques. This may change max_degree.
  if (use_dag) {
    assert(!directed); // must be undirected before orientation
    this->orientation();
  }
  VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);
  labels_frequency_.clear();
}

template<bool map_vertices, bool map_edges>
GraphT<map_vertices, map_edges>::~GraphT() {
  if (dst_list != NULL && dst_list != edges) delete [] dst_list;
  if constexpr (map_edges) munmap(edges, n_edges*sizeof(vidType));
  else custom_free(edges, n_edges);
  if constexpr (map_vertices) munmap(vertices, (n_vertices+1)*sizeof(eidType));
  else custom_free(vertices, n_vertices+1);
  if (vlabels != NULL) delete [] vlabels;
  if (elabels != NULL) delete [] elabels;
  if (features != NULL) delete [] features;
  if (src_list != NULL) delete [] src_list;
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::read_meta_info(std::string prefix, bool bipartite) {
  std::ifstream f_meta((prefix + ".meta.txt").c_str());
  assert(f_meta);
  int64_t nv = 0;
  if (bipartite) {
    f_meta >> n_vert0 >> n_vert1;
    nv = int64_t(n_vert0) + int64_t(n_vert1);
  } else f_meta >> nv;
  f_meta >> n_edges >> vid_size >> eid_size >> vlabel_size >> elabel_size
         >> max_degree >> feat_len >> num_vertex_classes >> num_edge_classes;
  assert(sizeof(vidType) == vid_size);
  assert(sizeof(eidType) == eid_size);
  assert(sizeof(vlabel_t) == vlabel_size);
  //assert(sizeof(elabel_t) == elabel_size);
  f_meta.close();
  assert(nv > 0 && n_edges > 0);
  if (vid_size == 4) assert(nv < 4294967295);
  n_vertices = nv;
  std::cout << "debug: |V| " << nv << " |E| " << n_edges << "\n";
}

template<> void GraphT<>::load_compressed_graph(std::string prefix) {
  // read meta information
  read_meta_info(prefix);
  assert(max_degree > 0 && max_degree < n_vertices);
  VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);

  // load row offsets
  read_file(prefix+".vertex.bin", vertices_compressed, n_vertices+1);
  //for (vidType v = 0; v < n_vertices+1; v++)
  //  std::cout << "rowptr[" << v << "]=" << vertices_compressed[v] << "\n";

  // load edges, i.e., column indices
  std::ifstream ifs;
  ifs.open(prefix+".edge.bin", std::ios::in | std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    std::cout << "open graph file failed!" << std::endl;
    exit(1);
  }
  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  ifs.read((char*) buffer.data(), size);
  edges_compressed.clear();
  vidType tmp = 0;
  for (size_t i = 0; i < buffer.size(); i++) {
    tmp <<= 8;
    tmp += buffer[i];
    if ((i + 1) % vid_size == 0) { // vidType has 4 bytes
      edges_compressed.push_back(tmp);
    }
  }
  if (size % vid_size) {
    int rem = size % vid_size;
    while (rem % vid_size)
      tmp <<= 8, rem++;
    edges_compressed.push_back(tmp);
  }
  ifs.close();
  is_compressed_ = true;
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

template<> vidType GraphT<>::decode_intervals(vidType v, CgrReader &decoder, vidType* ptr) {
  vidType num_neighbors = 0;
  // handle segmented intervals
  auto segment_cnt = decoder.decode_segment_cnt();
  // for each segment
  auto interval_offset = decoder.global_offset;
  for (vidType i = 0; i < segment_cnt; i++) {
    CgrReader cgrr(v, &edges_compressed[0], interval_offset);
    //IntervalSegmentHelper isHelper(v, decoder);
    IntervalSegmentHelper isHelper(v, cgrr);
    isHelper.decode_interval_cnt();
    auto num_intervals = isHelper.interval_cnt;
    // for each interval in the segment
    for (vidType j = 0; j < num_intervals; j++) {
      auto left = isHelper.get_interval_left();
      auto len = isHelper.get_interval_len();
      assert(left < n_vertices);
      assert(len < max_degree);
      for (vidType k = 0; k < len; k++) {
        ptr[num_neighbors++] = left+k;
      }
    }
    interval_offset += INTERVAL_SEGMENT_LEN;
    if (i == segment_cnt-1) // last segment
      decoder.global_offset = cgrr.global_offset;
    else
      decoder.global_offset += INTERVAL_SEGMENT_LEN;
  }
  return num_neighbors;
}

template<> vidType GraphT<>::decode_intervals(vidType v, CgrReader &decoder, VertexList &itv_begin, VertexList &itv_end) {
  vidType num_neighbors = 0;
  // handle segmented intervals
  auto segment_cnt = decoder.decode_segment_cnt();
  // for each segment
  auto interval_offset = decoder.global_offset;
  for (vidType i = 0; i < segment_cnt; i++) {
    CgrReader cgrr(v, &edges_compressed[0], interval_offset);
    //IntervalSegmentHelper isHelper(v, decoder);
    IntervalSegmentHelper isHelper(v, cgrr);
    isHelper.decode_interval_cnt();
    auto num_intervals = isHelper.interval_cnt;
    // for each interval in the segment
    for (vidType j = 0; j < num_intervals; j++) {
      auto left = isHelper.get_interval_left();
      auto len = isHelper.get_interval_len();
      assert(left < n_vertices);
      assert(len < max_degree);
      itv_begin.push_back(left);
      itv_end.push_back(left+len);
      num_neighbors += len;
    }
    interval_offset += INTERVAL_SEGMENT_LEN;
    if (i == segment_cnt-1) // last segment
      decoder.global_offset = cgrr.global_offset;
    else
      decoder.global_offset += INTERVAL_SEGMENT_LEN;
  }
  return num_neighbors;
}

template<> vidType GraphT<>::decode_residuals(vidType v, CgrReader &decoder, vidType offset, vidType *ptr) {
  vidType num_neighbors = offset;
  // handle segmented residuals
  auto segment_cnt = decoder.decode_segment_cnt();
  auto residual_offset = decoder.global_offset;
  for (vidType i = 0; i < segment_cnt; i++) {
    CgrReader cgrr(v, &edges_compressed[0], residual_offset);
    ResidualSegmentHelper rsHelper(v, cgrr);
    rsHelper.decode_residual_cnt();
    auto num_res = rsHelper.residual_cnt;
    // for each residual in the segment
    for (vidType j = 0; j < num_res; j++) {
      auto residual = rsHelper.get_residual();
      ptr[num_neighbors++] = residual;
    }
    residual_offset += RESIDUAL_SEGMENT_LEN;
    decoder.global_offset += RESIDUAL_SEGMENT_LEN;
  }
  return num_neighbors;
}

template<> vidType GraphT<>::decode_vertex(vidType v, vidType* ptr) {
  CgrReader decoder(v, &edges_compressed[0], vertices_compressed[v]);
  vidType num_neighbors = 0;
#if USE_INTERVAL
  num_neighbors = decode_intervals(v, decoder, ptr);
#endif
  num_neighbors = decode_residuals(v, decoder, num_neighbors, ptr);
  return num_neighbors;
}

template<> void GraphT<>::decode_vertex(vidType v, VertexSet& adj, bool ordered) {
  vidType deg = decode_vertex(v, adj.data());
  adj.adjust_size(deg);
  if (ordered) adj.sort();
}

template<> void GraphT<>::decompress() {
  Timer t;
  t.Start();
  vertices = new eidType[n_vertices+1];
  edges = new vidType[n_edges];
  vertices[0] = 0;
  eidType offset = 0;
  //auto ptr = &edges_compressed[0];
  for (vidType v = 0; v < n_vertices; v++) {
    auto num = decode_vertex(v, &edges[offset]);
    offset += num;
    vertices[v+1] = offset;
    std::sort(edges+vertices[v], edges+offset);
  }
  t.Stop();
  std::cout << "Graph decompressed time: " << t.Seconds() << "\n";
}
 
template<bool map_vertices, bool map_edges>
VertexSet GraphT<map_vertices, map_edges>::N(vidType vid) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  eidType begin = vertices[vid], end = vertices[vid+1];
  if (begin > end || end > n_edges) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(edges + begin, end - begin, vid);
}

template<> void GraphT<>::sort_neighbors() {
  std::cout << "Sorting the neighbor lists (used for pattern mining)\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    auto begin = edge_begin(v);
    auto end = edge_end(v);
    std::sort(edges+begin, edges+end);
  }
}

template<> void GraphT<>::sort_and_clean_neighbors() {
  std::cout << "Sorting the neighbor lists and remove selfloops and redundent edges (used for pattern mining)\n";
  std::vector<vidType> degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    auto begin = edge_begin(v);
    auto end = edge_end(v);
    std::sort(edges+begin, edges+end);
    eidType i = 0;
    for (auto u : N(v)) {
      if (u == v) continue;
      if (i>0 && u == N(v, i-1)) continue;
      degrees[v] ++;
      i++;
    }
  }
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  parallel_prefix_sum<vidType,eidType>(degrees, new_vertices);
  n_edges = new_vertices[n_vertices];
  std::cout << "|E| after clean: " << n_edges << "\n";
  vidType *new_edges = custom_alloc_global<vidType>(n_edges);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v ++) {
    auto begin = new_vertices[v];
    eidType i = 0;
    for (auto u : N(v)) {
      if (u == v) continue;
      if (i>0 && u == N(v, i-1)) continue;
      new_edges[begin+i] = u;
      i++;
    }
  }
  delete [] vertices;
  delete [] edges;
  vertices = new_vertices;
  edges = new_edges;
}

template<> void GraphT<>::write_to_file(std::string outfilename, bool v, bool e, bool vl, bool el) {
  if (v) {
    std::ofstream outfile((outfilename+".vertex.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(vertices), (n_vertices+1)*sizeof(eidType));
    outfile.close();
  }

  if (e) {
    std::ofstream outfile1((outfilename+".edge.bin").c_str(), std::ios::binary);
    if (!outfile1) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile1.write(reinterpret_cast<const char*>(edges), n_edges*sizeof(vidType));
    outfile1.close();
  }

  if (vl && vlabels) {
    std::ofstream outfile((outfilename+".vlabel.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(&vlabels[0]), n_vertices*sizeof(vlabel_t));
    outfile.close();
  }

  if (el && elabels) {
    std::ofstream outfile((outfilename+".elabel.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(&elabels[0]), n_edges*sizeof(elabel_t));
    outfile.close();
  }
}
 
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::build_reverse_graph() {
  std::vector<VertexList> reverse_adj_lists(n_vertices);
  for (vidType v = 0; v < n_vertices; v++) {
    for (auto u : N(v)) {
      reverse_adj_lists[u].push_back(v);
    }
  }
  reverse_vertices = custom_alloc_global<eidType>(n_vertices+1);
  reverse_vertices[0] = 0;
  for (vidType i = 1; i < n_vertices+1; i++) {
    auto degree = reverse_adj_lists[i-1].size();
    reverse_vertices[i] = reverse_vertices[i-1] + degree;
  }
  reverse_edges = custom_alloc_global<vidType>(n_edges);
  //#pragma omp parallel for
  for (vidType i = 0; i < n_vertices; i++) {
    auto begin = reverse_vertices[i];
    std::copy(reverse_adj_lists[i].begin(), 
        reverse_adj_lists[i].end(), &reverse_edges[begin]);
  }
  for (auto adjlist : reverse_adj_lists) adjlist.clear();
  reverse_adj_lists.clear();
}

template<> VertexSet GraphT<>::out_neigh(vidType vid, vidType offset) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  auto begin = vertices[vid];
  auto end = vertices[vid+1];
  if (begin > end) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(edges + begin + offset, end - begin, vid);
}

// TODO: fix for directed graph
template<> VertexSet GraphT<>::in_neigh(vidType vid) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  auto begin = reverse_vertices[vid];
  auto end = reverse_vertices[vid+1];
  if (begin > end) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(reverse_edges + begin, end - begin, vid);
}
 
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::allocateFrom(vidType nv, eidType ne) {
  n_vertices = nv;
  n_edges    = ne;
  vertices = new eidType[nv+1];
  edges = new vidType[ne];
  vertices[0] = 0;
}
 
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::compute_max_degree() {
  std::cout << "computing the maximum degree\n";
  Timer t;
  t.Start();
  #pragma omp parallel for reduction(max:max_degree)
  for (vidType v = 0; v < n_vertices; v++) {
    auto deg = this->get_degree(v);
    if (deg > max_degree) max_degree = deg;
  }
  /*
  std::vector<vidType> degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = vertices[v+1] - vertices[v];
  }
  max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  */
  t.Stop();
  std::cout << "maximum degree: " << max_degree << "\n";
  std::cout << "Time computing the maximum degree: " << t.Seconds() << " sec\n";
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::orientation() {
  std::cout << "Orientation enabled, using DAG\n";
  Timer t;
  t.Start();
  std::vector<vidType> degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = this->get_degree(v);
  }
  std::vector<vidType> new_degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_degrees[src]++;
      }
    }
  }
  max_degree = *(std::max_element(new_degrees.begin(), new_degrees.end()));
  eidType *old_vertices = vertices;
  vidType *old_edges = edges;
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  //prefix_sum<vidType,eidType>(new_degrees, new_vertices);
  parallel_prefix_sum<vidType,eidType>(new_degrees, new_vertices);
  auto num_edges = new_vertices[n_vertices];
  vidType *new_edges = custom_alloc_global<vidType>(num_edges);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    auto begin = new_vertices[src];
    eidType offset = 0;
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_edges[begin+offset] = dst;
        offset ++;
      }
    }
  }
  vertices = new_vertices;
  edges = new_edges;
  custom_free<eidType>(old_vertices, n_vertices);
  custom_free<vidType>(old_edges, n_edges);
  n_edges = num_edges;
  t.Stop();
  std::cout << "Time on generating the DAG: " << t.Seconds() << " sec\n";
}

template<> void GraphT<>::print_neighbors(vidType v) const {
  eidType begin = vertices[v], end = vertices[v+1];
  std::cout << "[ ";
  for (auto e = begin; e != end; e++) {
    if (elabels != NULL)
      std::cout << "<";
    std::cout << getEdgeDst(e) << " ";
    if (elabels != NULL)
      std::cout << getEdgeData(e) << "> ";
  }
  std::cout << "]";
}

template<> void GraphT<>::print_graph() const {
  std::cout << "Printing the graph: \n";
  for (vidType n = 0; n < n_vertices; n++) {
    eidType begin = vertices[n], end = vertices[n+1];
    std::cout << "vertex " << n << ": degree = " << this->get_degree(n) 
      << " edge range: [" << begin << ", " << end << ")"
      << " edgelist = "; 
    print_neighbors(n);
    std::cout << "\n";
  }
}

template<> eidType GraphT<>::init_edgelist(bool sym_break, bool ascend) {
  Timer t;
  t.Start();
  if (nnz != 0) return nnz; // already initialized
  nnz = E();
  if (sym_break) nnz = nnz/2;
  sizes.resize(V());
  src_list = new vidType[nnz];
  if (sym_break) dst_list = new vidType[nnz];
  else dst_list = edges;
  size_t i = 0;
  for (vidType v = 0; v < V(); v ++) {
    for (auto u : N(v)) {
      if (u == v) continue; // no selfloops
      if (ascend) {
        if (sym_break && v > u) continue;  
      } else {
        if (sym_break && v < u) break;  
      }
      src_list[i] = v;
      if (sym_break) dst_list[i] = u;
      sizes[v] ++;
      i ++;
    }
  }
  //assert(i == nnz);
  t.Stop();
  std::cout << "Time on generating the edgelist: " << t.Seconds() << " sec\n";
  return nnz;
}

template<> bool GraphT<>::binary_search(vidType key, eidType begin, eidType end) const {
  auto l = begin;
  auto r = end-1;
  while (r >= l) { 
    auto mid = l + (r - l) / 2;
    auto value = getEdgeDst(mid);
    if (value == key) return true;
    if (value < key) l = mid + 1; 
    else r = mid - 1; 
  } 
  return false;
}

template<> bool GraphT<>::is_connected(vidType v, vidType u) const {
  auto v_deg = this->get_degree(v);
  auto u_deg = this->get_degree(u);
  bool found;
  if (v_deg < u_deg) {
    found = binary_search(u, edge_begin(v), edge_end(v));
  } else {
    found = binary_search(v, edge_begin(u), edge_end(u));
  }
  return found;
}

template<> bool GraphT<>::is_connected(std::vector<vidType> sg) const {
  return false;
}

template<> vidType GraphT<>::intersect_num_compressed(VertexSet& vs, vidType u, vidType up) {
  vidType num = 0;
  Timer t;
  t.Start();
  CgrReader u_decoder(u, &edges_compressed[0], vertices_compressed[u]);
#if USE_INTERVAL
  VertexList u_begins, u_ends;
  auto u_deg_itv = decode_intervals(u, u_decoder, u_begins, u_ends);
  int u_size = u_begins.size();
#endif
  VertexSet u_residuals(u);
  auto u_deg_res = decode_residuals(u, u_decoder, 0, u_residuals.data());
  u_residuals.adjust_size(u_deg_res);
  int v_size = vs.size();
  t.Stop();
  timers[DECOMPRESS] += t.Seconds();

  t.Start();
#if USE_INTERVAL
  // compare vs and u_itv
  int idx_l = 0, idx_r = 0;
  while (idx_l < v_size && idx_r < u_size) {
    auto v = vs[idx_l];
    if (v >= up) break;
    auto u_begin = u_begins[idx_r];
    if (u_begin >= up) break;
    if (v < u_begin) {
      idx_l++;
      continue;
    }
    auto u_end = u_ends[idx_r];
    if (v >= u_end) {
      if (v == u_end) idx_l++;
      idx_r++;
      continue;
    }
    if (v >= u_begin && v < u_end) {
      num++;
      idx_l++;
    }
  }
#endif
  // compare vs and u_res
  num += intersection_num(vs, u_residuals, up);
  t.Stop();
  timers[SETOPS] += t.Seconds();
  return num;
}

template<> vidType GraphT<>::intersect_num_compressed(vidType v, vidType u, vidType up) {
  vidType num = 0;
  VertexList v_begins, v_ends;
  VertexList u_begins, u_ends;
  VertexSet v_residuals(v);
  VertexSet u_residuals(u);

  CgrReader v_decoder(v, &edges_compressed[0], vertices_compressed[v]);
  auto v_deg_itv = decode_intervals(v, v_decoder, v_begins, v_ends);
  auto v_deg_res = decode_residuals(v, v_decoder, 0, v_residuals.data());
  v_residuals.adjust_size(v_deg_res);

  CgrReader u_decoder(u, &edges_compressed[0], vertices_compressed[u]);
  auto u_deg_itv = decode_intervals(u, u_decoder, u_begins, u_ends);
  auto u_deg_res = decode_residuals(u, u_decoder, 0, u_residuals.data());
  u_residuals.adjust_size(u_deg_res);

  int v_size = v_begins.size();
  int u_size = u_begins.size();

  // compare v_itv and u_itv
  int idx_l = 0, idx_r = 0;
  while (idx_l < v_size && idx_r < u_size) {
    auto v_begin = v_begins[idx_l];
    auto v_end = v_ends[idx_l];
    auto u_begin = u_begins[idx_r];
    auto u_end = u_ends[idx_r];
    assert(v_end > v_begin);
    assert(u_end > u_begin);
    if (v_begin >= up || u_begin >= up) break;
    if (v_begin >= u_end) {
      idx_r++;
      continue;
    }
    if (u_begin >= v_end) {
      idx_l++;
      continue;
    }
    if (v_end >= u_end) idx_r++;
    if (v_end <= u_end) idx_l++;
    num += std::min(up, std::min(v_end, u_end)) - std::max(v_begin, u_begin);
  }
  // compare v_itv and u_res
  idx_l = 0, idx_r = 0;
  int u_res_size = u_residuals.size();
  while (idx_l < v_size && idx_r < u_res_size) {
    auto u_res = u_residuals[idx_r];
    if (u_res >= up) break;
    auto v_begin = v_begins[idx_l];
    if (v_begin >= up) break;
    if (u_res < v_begin) {
      idx_r++;
      continue;
    }
    auto v_end = v_ends[idx_l];
    if (u_res >= v_end) {
      if (u_res == v_end) idx_r++;
      idx_l++;
      continue;
    }
    if (u_res >= v_begin && u_res < v_end) {
      num++;
      idx_r++;
    }
  }
  // compare v_res and u_itv
  for (auto v_res : v_residuals) {
    if (v_res >= up) break;
    for (int i = 0; i < u_size; i++) {
      auto u_begin = u_begins[i];
      if (u_begin >= up) break;
      if (v_res < u_begin) continue;
      auto u_end = u_ends[i];
      if (v_res < u_end) {
        num++;
        break;
      }
    }
  }
  // compare v_res and u_res
  num += intersection_num(v_residuals, u_residuals, up);
  return num;
}

template<> vidType GraphT<>::intersect_num(vidType v, vidType u) {
  return N(v).get_intersect_num(N(u));
}

template<> vidType GraphT<>::intersect_num(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = this->get_degree(v);
  vidType u_size = this->get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    vidType a = v_ptr[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) num++;
  }
  return num;
}

template<> vidType GraphT<>::intersect_num(VertexSet& vs, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = this->get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    vidType a = vs[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) num++;
  }
  return num;
}

template<> vidType GraphT<>::intersect_set(vidType v, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = this->get_degree(v);
  vidType u_size = this->get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    vidType a = v_ptr[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  return num;
}

template<> vidType GraphT<>::intersect_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = this->get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    vidType a = vs[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  return num;
}

template<> vidType GraphT<>::difference_num_edgeinduced(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType* v_ptr = &edges[vertices[v]];
  for (vidType i = 0; i < this->get_degree(v); i ++) {
    auto w = v_ptr[i];
    if (w != u && vlabels[w] == label) num++;
  }
  return num;
}

template<> vidType GraphT<>::difference_num_edgeinduced(VertexSet& vs, vidType u, vlabel_t label) {
  vidType num = 0;
  for (auto w : vs)
    if (w != u && vlabels[w] == label) num++;
  return num;
}

template<> vidType GraphT<>::difference_set_edgeinduced(vidType v, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType* v_ptr = &edges[vertices[v]];
  for (vidType i = 0; i < this->get_degree(v); i ++) {
    auto w = v_ptr[i];
    if (w != u && vlabels[w] == label) {
      result.add(w);
      num++;
    }
  }
  return num;
}

template<> vidType GraphT<>::difference_set_edgeinduced(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  for (auto w : vs) {
    if (w != u && vlabels[w] == label) {
      result.add(w);
      num++;
    }
  }
  return num;
}

template<> vidType GraphT<>::difference_num(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = this->get_degree(v);
  vidType u_size = this->get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    auto a = v_ptr[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) num++;
  }
  while (idx_l < v_size) {
    auto a = v_ptr[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label)
      num ++;
  }
  return num;
}

template<> vidType GraphT<>::difference_num(VertexSet& vs, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = this->get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    auto a = vs[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) num++;
  }
  while (idx_l < vs.size()) {
    auto a = vs[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label)
      num ++;
  }
  return num;
}

template<> vidType GraphT<>::difference_set(vidType v, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = this->get_degree(v);
  vidType u_size = this->get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    auto a = v_ptr[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  while (idx_l < v_size) {
    auto a = v_ptr[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label) {
      result.add(a);
      num ++;
    }
  }
  return num;
}

template<> vidType GraphT<>::difference_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = this->get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    auto a = vs[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  while (idx_l < vs.size()) {
    auto a = vs[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label) {
      result.add(a);
      num ++;
    }
  }
  return num;
}

#pragma omp declare reduction(vec_plus : std::vector<vidType> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<vidType>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
template<> void GraphT<>::computeLabelsFrequency() {
  labels_frequency_.resize(num_vertex_classes+1);
  std::fill(labels_frequency_.begin(), labels_frequency_.end(), 0);
  //max_label = int(*std::max_element(vlabels, vlabels+size()));
  #pragma omp parallel for reduction(max:max_label)
  for (vidType i = 0; i < size(); ++i) {
    max_label = max_label > vlabels[i] ? max_label : vlabels[i];
  }
  #pragma omp parallel for reduction(vec_plus:labels_frequency_)
  for (vidType v = 0; v < size(); ++v) {
    int label = int(this->get_vlabel(v));
    assert(label <= num_vertex_classes);
    labels_frequency_[label] += 1;
  }
  max_label_frequency_ = vidType(*std::max_element(labels_frequency_.begin(), labels_frequency_.end()));
  //std::cout << "max_label = " << max_label << "\n";
  //std::cout << "max_label_frequency_ = " << max_label_frequency_ << "\n";
  //for (size_t i = 0; i < labels_frequency_.size(); ++i)
  //  std::cout << "label " << i << " vertex frequency: " << labels_frequency_[i] << "\n";
}

template<> void GraphT<>::BuildReverseIndex() {
  if (labels_frequency_.empty()) computeLabelsFrequency();
  int nl = num_vertex_classes;
  if (max_label == num_vertex_classes) nl += 1;
  reverse_index_.resize(size());
  reverse_index_offsets_.resize(nl+1);
  reverse_index_offsets_[0] = 0;
  vidType total = 0;
  for (int i = 0; i < nl; ++i) {
    total += labels_frequency_[i];
    reverse_index_offsets_[i+1] = total;
    //std::cout << "label " << i << " frequency: " << labels_frequency_[i] << "\n";
  }
  std::vector<eidType> start(nl);
  for (int i = 0; i < nl; ++i) {
    start[i] = reverse_index_offsets_[i];
    //std::cout << "label " << i << " start: " << start[i] << "\n";
  }
  for (vidType i = 0; i < size(); ++i) {
    auto vl = vlabels[i];
    reverse_index_[start[vl]++] = i;
  }
}

template<> int GraphT<>::get_frequent_labels(int threshold) {
  assert(threshold > 0);
  int num = 0;
  for (size_t i = 0; i < labels_frequency_.size(); ++i)
    if (labels_frequency_[i] > vidType(threshold))
      num++;
  return num;
}

template<> bool GraphT<>::is_freq_vertex(vidType v, int threshold) {
  assert(threshold > 0);
  assert(v >= 0 && v < size());
  auto label = int(vlabels[v]);
  assert(label <= num_vertex_classes);
  if (labels_frequency_[label] >= vidType(threshold)) return true;
  return false;
}

// NLF: neighborhood label frequency
template<> void GraphT<>::BuildNLF() {
  //std::cout << "Building NLF map for the data graph\n";
  nlf_.resize(size());
  #pragma omp parallel for
  for (vidType v = 0; v < size(); ++v) {
    for (auto u : N(v)) {
      auto vl = this->get_vlabel(u);
      if (nlf_[v].find(vl) == nlf_[v].end())
        nlf_[v][vl] = 0;
      nlf_[v][vl] += 1;
    }
  }
}

template<> void GraphT<>::print_meta_data() const {
  std::cout << "|V|: " << n_vertices << ", |E|: " << n_edges << ", Max Degree: " << max_degree << "\n";
  if (num_vertex_classes > 0) {
    std::cout << "vertex-|\u03A3|: " << num_vertex_classes;
    if (!labels_frequency_.empty()) 
      std::cout << ", Max Label Frequency: " << max_label_frequency_;
    std::cout << "\n";
  } else {
    std::cout  << "This graph does not have vertex labels\n";
  }
  if (num_edge_classes > 0) {
    std::cout << "edge-|\u03A3|: " << num_edge_classes << "\n";
  } else {
    std::cout  << "This graph does not have edge labels\n";
  }
  if (feat_len > 0) {
    std::cout << "Vertex feature vector length: " << feat_len << "\n";
  } else {
    std::cout  << "This graph has no input vertex features\n";
  }
}

template <> void GraphT<>::computeKCore() {
  int nv = size();
  int md = get_max_degree();
  std::vector<int> vertices(nv);          // Vertices sorted by degree.
  std::vector<int> position(nv);          // The position of vertices in vertices array.
  std::vector<int> degree_bin(md+1, 0);   // Degree from 0 to max_degree.
  std::vector<int> offset(md+1);          // The offset in vertices array according to degree.
  for (int i = 0; i < nv; ++i) {
    int degree = this->get_degree(i);
    core_table[i] = degree;
    degree_bin[degree] += 1;
  }
  int start = 0;
  for (int i = 0; i < md+1; ++i) {
    offset[i] = start;
    start += degree_bin[i];
  }
  for (int i = 0; i < nv; ++i) {
    int degree = this->get_degree(i);
    position[i] = offset[degree];
    vertices[position[i]] = i;
    offset[degree] += 1;
  }
  for (int i = md; i > 0; --i) {
    offset[i] = offset[i - 1];
  }
  offset[0] = 0;
  for (int i = 0; i < nv; ++i) {
    int v = vertices[i];
    for (vidType j = 0; j < this->get_degree(v); ++j) {
      int u = N(v, j);
      if (core_table[u] > core_table[v]) {
        // Get the position and vertex which is with the same degree
        // and at the start position of vertices array.
        int cur_degree_u = core_table[u];
        int position_u = position[u];
        int position_w = offset[cur_degree_u];
        int w = vertices[position_w];
        if (u != w) {
          // Swap u and w.
          position[u] = position_w;
          position[w] = position_u;
          vertices[position_u] = w;
          vertices[position_w] = u;
        }
        offset[cur_degree_u] += 1;
        core_table[u] -= 1;
      }
    }
  }
}

template<> void GraphT<>::buildCoreTable() {
  core_table.resize(size(), 0);
  computeKCore();
  for (vidType i = 0; i < size(); ++i) {
    if (core_table[i] > 1) {
      core_length_ += 1;
    }
  }
  //for (int v = 0; v < size(); v++)
  //  std::cout << "v_" << v << " core value: " << core_table[v] << "\n";
}

template class GraphT<false, false>;
template class GraphT<true, false>;
//template class GraphT<true, true>;
