#pragma once
#include "VertexSet.h"

using namespace std;

template <bool map_vertices=false, bool map_edges=false>
class GraphT {
protected:
  std::string name_;            // name of the graph
  std::string inputfile_path;   // input file path of the graph
  std::string inputfile_prefix; // input file prefix of the graph
  bool is_directed_;            // is it a directed graph?
  bool is_bipartite_;           // is it a bipartite graph?
  bool is_compressed_;          // is it a compressed graph?
  bool has_reverse;             // has reverse/incoming edges maintained
  vidType max_degree;           // maximun degree
  vidType n_vertices;           // number of vertices
  eidType n_edges;              // number of edges
  eidType nnz;                  // number of edges in COO format (may be halved due to orientation)
  vidType max_label_frequency_; // maximum label frequency
  vidType n_vert0;              // number of type 0 vertices for bipartite graph
  vidType n_vert1;              // number of type 1 vertices for bipartite graph
  int max_label;                // maximum label
  int feat_len;                 // vertex feature vector length: '0' means no features
  int num_vertex_classes;       // number of distinct vertex labels: '0' means no vertex labels
  int num_edge_classes;         // number of distinct edge labels: '0' means no edge labels
  int core_length_;
  int vid_size, eid_size, vlabel_size, elabel_size; // number of bytes for vid, eid, vlabel, elabel
  vidType degree_threshold;

  vidType *edges;               // column indices of CSR format
  eidType *vertices;            // row pointers of CSR format
  vidType *reverse_edges;       // reverse column indices of CSR format
  eidType *reverse_vertices;    // reverse row pointers of CSR format
  vlabel_t *vlabels;            // vertex labels
  elabel_t *elabels;            // edge labels
  feat_t *features;             // vertex features; one feature vector per vertex
  vidType *src_list, *dst_list; // source and destination vertices of COO format
  std::vector<int> core_table;  // coreness for each vertex
  VertexList labels_frequency_; // vertex count of each label
  VertexList sizes;             // neighbor count of each source vertex in the edgelist
  VertexList reverse_index_;    // indices to vertices grouped by vertex label
  eidType *vertices_compressed; // row pointers of the compressed format
  std::vector<nlf_map> nlf_;    // neighborhood label frequency
  std::vector<eidType> reverse_index_offsets_; // pointers to each vertex group
  std::vector<uint32_t> edges_compressed;      // compressed edgelist
  std::vector<vidType> degrees; 

public:
  GraphT(std::string prefix,
         bool use_dag = false,
         bool directed = false,
         bool use_vlabel = false,
         bool use_elabel = false,
         bool need_reverse = false,
         bool bipartite = false,
         bool partitioned = false);
  GraphT(bool directed, bool bipartite) :
            name_(""), 
            is_directed_(directed),
            is_bipartite_(bipartite),
            is_compressed_(0), has_reverse(0),
            max_degree(0), n_vertices(0),
            n_edges(0), nnz(0), 
            max_label_frequency_(0),
            n_vert0(0), n_vert1(0),
            max_label(0), feat_len(0), 
            num_vertex_classes(0),
            num_edge_classes(0),
            core_length_(0),
            vid_size(4), eid_size(8),
            vlabel_size(0), elabel_size(0),
            degree_threshold(32),
            edges(NULL), vertices(NULL),
            reverse_edges(NULL), reverse_vertices(NULL),
            vlabels(NULL), elabels(NULL),
            features(NULL),
            src_list(NULL), dst_list(NULL) { }
  GraphT(vidType nv, eidType ne) : GraphT() { allocateFrom(nv, ne); }
  GraphT() : GraphT(false, false) {}
  ~GraphT();
  GraphT(const GraphT &)=delete;
  GraphT& operator=(const GraphT &)=delete;

  void load_graph(std::string prefix, 
                  bool use_dag = false, bool use_vlabel = false, 
                  bool use_elabel = false, bool need_reverse = false, 
                  bool partitioned = false);
  void load_graph_data(std::string prefix, 
                       bool use_dag = false, bool use_vlabel = false, 
                       bool use_elabel = false, bool need_reverse = false);
  void deallocate();

  // get methods for graph meta information
  vidType V() const { return n_vertices; }
  eidType E() const { return n_edges; }
  vidType V(int type) const { if (type == 0) return n_vert0; else return n_vert1; }
  eidType get_num_tasks() const { return nnz; }
  vidType num_vertices() const { return n_vertices; }
  eidType num_edges() const { return n_edges; }
  std::string get_name() const { return name_; }
  std::string get_inputfile_path() const { return inputfile_path; }
  std::string get_inputfile_prefix() const { return inputfile_prefix; }
  bool is_directed() const { return is_directed_; }
  bool is_bipartite() const { return is_bipartite_; }
  bool has_reverse_graph() const { return has_reverse; }
  vidType get_max_degree() const { return max_degree; }

  // get methods for graph topology information
  vidType read_degree(vidType v) const { return degrees[v]; }
  const vidType* get_degrees_ptr() const { return degrees.data(); }
  vidType get_degree_threshold() const { return degree_threshold; }
  vidType get_degree(vidType v) const { return vertices[v+1] - vertices[v]; }
  vidType out_degree(vidType v) const { return vertices[v+1] - vertices[v]; }
  eidType edge_begin(vidType v) const { return vertices[v]; }
  eidType edge_end(vidType v) const { return vertices[v+1]; }
  vidType* adj_ptr(vidType v) const { return &edges[vertices[v]]; }
  vidType N(vidType v, vidType n) const { return edges[vertices[v]+n];} // get the n-th neighbor of v
  VertexSet N(vidType v) const;                                         // get the neighbor list of vertex v
  eidType get_eid(vidType v, vidType n) const { return vertices[v]+n;}  // get the edge id of the n-th edge of v
  eidType* rowptr() { return vertices; }             // get row pointers array
  vidType* colidx() { return edges; }                // get column indices array
  const eidType* rowptr() const { return vertices; } // get row pointers array
  const vidType* colidx() const { return edges; }    // get column indices array
  eidType* out_rowptr() { return vertices; }         // get row pointers array
  vidType* out_colidx() { return edges; }            // get column indices array
  eidType* in_rowptr() { return reverse_vertices; }  // get incoming row pointers array
  vidType* in_colidx() { return reverse_edges; }     // get incoming column indices array
  bool is_connected(vidType v, vidType u) const;     // is vertex v and u connected by an edge
  bool is_connected(std::vector<vidType> sg) const;  // is the subgraph sg a connected one
  VertexSet out_neigh(vidType v, vidType off = 0) const; // get the outgoing neighbor list of vertex v
  VertexSet in_neigh(vidType v) const;               // get the ingoing neighbor list of vertex v
  void build_reverse_graph();
 
  // Galois compatible APIs
  vidType size() const { return n_vertices; }
  eidType sizeEdges() const { return n_edges; }
  vidType getEdgeDst(eidType e) const { return edges[e]; } // get target vertex of the edge e
  vlabel_t getData(vidType v) const { return vlabels[v]; }
  vlabel_t getVertexData(vidType v) const { return vlabels[v]; }
  elabel_t getEdgeData(eidType e) const { return elabels[e]; }
  void fixEndEdge(vidType vid, eidType row_end) { vertices[vid + 1] = row_end; }
  void allocateFrom(vidType nv, eidType ne);
  void constructEdge(eidType eid, vidType dst) { edges[eid] = dst; }

  // get methods for labels and coreness
  vlabel_t get_vlabel(vidType v) const { return vlabels[v]; }
  elabel_t get_elabel(eidType e) const { return elabels[e]; }
  elabel_t get_elabel(vidType v, vidType n) const { return elabels[vertices[v]+n]; } // get the label of the n-th edge of v
  int get_vertex_classes() { return num_vertex_classes; } // number of distinct vertex labels
  int get_edge_classes() { return num_edge_classes; } // number of distinct edge labels
  int get_frequent_labels(int threshold);
  int get_max_label() { return max_label; }
  vlabel_t* getVlabelPtr() { return vlabels; }
  elabel_t* getElabelPtr() { return elabels; }
  vlabel_t* get_vlabel_ptr() { return vlabels; }
  elabel_t* get_elabel_ptr() { return elabels; }
  bool has_label() { return vlabels != NULL || elabels != NULL; }
  bool has_vlabel() { return vlabels != NULL; }
  bool has_elabel() { return elabels != NULL; }
  int getCoreValue(const vidType vid) const { return core_table[vid]; }
  int get2CoreSize() const { return core_length_; }

  // edgelist or COO
  vidType* get_src_ptr() { return &src_list[0]; }
  vidType* get_dst_ptr() { return &dst_list[0]; }
  vidType get_src(eidType eid) { return src_list[eid]; }
  vidType get_dst(eidType eid) { return dst_list[eid]; }
  std::vector<vidType> get_sizes() const { return sizes; }
  eidType init_edgelist(bool sym_break = false, bool ascend = false);

  // build auxiliary structures for vertex label frequency
  void BuildNLF();               // NLF: neighborhood label frequency
  void BuildReverseIndex();      // reverse index
  void computeLabelsFrequency();
  void buildCoreTable();
  void computeKCore();
  void sort_neighbors(); // sort the neighbor lists
  void sort_and_clean_neighbors(std::string outfile = ""); // sort the neighbor lists and remove selfloops and redundant edges
  void symmetrize(); // symmetrize a directed graph
  void write_to_file(std::string outfilename, bool v=1, bool e=1, bool vl=0, bool el=0);
  bool is_freq_vertex(vidType v, int minsup);
  vidType get_max_label_frequency() const { return max_label_frequency_; }
  const nlf_map* getVertexNLF(const vidType id) const { return &nlf_[id]; }
  vidType *get_label_freq_ptr() { return labels_frequency_.data(); }
  vidType getLabelsFrequency(vlabel_t label) const { return labels_frequency_.at(label); }
  const vidType* getVerticesByLabel(vlabel_t vl, vidType& count) const {
    auto start = reverse_index_offsets_[vl];
    count = reverse_index_offsets_[vl+1] - start;
    return &reverse_index_[start];
  }

  void compute_max_degree();
  void orientation(std::string outfile = ""); // edge orientation: convert the graph from undirected to directed
  void degree_histogram(int bin_width = 100, std::string outfile = ""); // compute the degree distribution
  vidType intersect_num(vidType v, vidType u);
  vidType intersect_num(vidType v, vidType u, vlabel_t label);
  vidType intersect_num(VertexSet& vs, vidType u, vlabel_t label);
  vidType intersect_set(vidType v, vidType u, VertexSet& result);
  vidType intersect_set(vidType v, vidType u, vlabel_t label, VertexSet& result);
  vidType intersect_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result);
  vidType difference_num(vidType v, vidType u);
  vidType difference_num(vidType v, vidType u, vlabel_t label);
  vidType difference_num(VertexSet& vs, vidType u, vlabel_t label);
  vidType difference_set(vidType v, vidType u, VertexSet& result);
  vidType difference_set(vidType v, vidType u, vlabel_t label, VertexSet& result);
  vidType difference_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result);
  vidType difference_num_edgeinduced(vidType v, vidType u, vlabel_t label);
  vidType difference_num_edgeinduced(VertexSet& vs, vidType u, vlabel_t label);
  vidType difference_set_edgeinduced(vidType v, vidType u, vlabel_t label, VertexSet& result);
  vidType difference_set_edgeinduced(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result);

  // graph compression
  bool is_compressed() const { return is_compressed_; }
  bool is_compressed_only() const { return (vertices == NULL) && is_compressed_; }
  void print_compressed_colidx();
  void load_compressed_graph(std::string prefix, std::string scheme = "", bool permutated = false);
  void load_row_pointers(std::string prefix);
  void decompress(std::string scheme = "cgr");
  void decode_vertex_cgr(vidType v, VertexSet &adj, bool ordered = 1);
  vidType decode_vertex_cgr(vidType v, vidType* out_ptr);
  vidType decode_vertex_vbyte(vidType v, vidType* out_ptr, std::string scheme);
  vidType decode_vertex_hybrid(vidType v, vidType* out_ptr, std::string scheme, bool use_segment, bool reverse);
  vidType decode_vertex_unary(vidType v, vidType* out_ptr);
  void decode_vertex_unary(vidType v, vidType* out_ptr, vidType degree);
  void decode_vertex_unary_segmented(vidType v, vidType* out, vidType degree);
  void set_degree_threshold(vidType deg) { degree_threshold = deg; }
  VertexSet N_cgr(vidType v, bool use_segment=false);                        // get the CGR compressed neighbor list of vertex v
  VertexSet N_vbyte(vidType v, std::string scheme);                          // get the VByte compressed neighbor list of vertex v
  VertexSet N_hybrid(vidType v, std::string scheme, bool use_segment=false); // get the hybrid compressed neighbor list of vertex v
  VertexSet get_interval_neighbors(vidType v);                               // get the interval neighbors in a CGR graph
  const eidType* rowptr_compressed() const { return vertices_compressed; }   // get compressed row pointers array
  const uint32_t* colidx_compressed() const { return &edges_compressed[0]; } // get compressed column indices array
  size_t get_compressed_colidx_length() const { return edges_compressed.size(); }
  vidType intersect_num_compressed(vidType v, vidType u);
  vidType intersect_num_compressed(vidType v, vidType u, vidType up);
  vidType intersect_num_compressed(VertexSet& vs, vidType u);
  vidType intersect_num_compressed(VertexSet& vs, vidType u, vidType up);

  // print graph information
  void print_meta_data() const;
  void print_graph() const;
  void print_neighbors(vidType v) const;

 protected:
  void read_meta_info(std::string prefix);
  bool binary_search(vidType key, eidType begin, eidType end) const;
};

typedef GraphT<false,false> Graph;
typedef GraphT<false,false> InMemGraph;
typedef GraphT<false,true>  SemiOutOfCoreGraph;
typedef GraphT<true,true>   OutOfCoreGraph;
typedef GraphT<false,false> BipartiteGraph;
