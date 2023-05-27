#pragma once
#include "global.h"
#include "timer.h"
#include "../utils.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifdef ENABLE_GPU
#define SUBGRAPH_SIZE (1024*128)
#define RANGE_WIDTH (512)
#else
#define SUBGRAPH_SIZE (1024*128)
#define RANGE_WIDTH (512)
#endif

class LearningGraph {
protected:
  bool is_device;
  bool partitioned;
  index_t num_vertices_;
  index_t num_edges_;
  index_t max_degree;
  index_t vdata_size;
  index_t edata_size;
  index_t gpu_vsize;
  index_t gpu_esize;

  index_t *rowptr_;
  index_t *colidx_;
  vdata_t *vertex_data_;
  edata_t *edge_data_;

  // for GPU
  index_t* d_rowptr_;
  index_t* d_colidx_;
  vdata_t* d_vertex_data_;
  edata_t* d_edge_data_;
  edata_t* d_trans_edge_data_;
  //index_t* d_degrees_;

  // for CSR segmenting
  int num_subgraphs;
  int num_ranges;
  index_t* nvs_of_subgraphs;     // nv of each subgraph
  index_t* nes_of_subgraphs;     // ne of each subgraph
  index_t** rowptr_blocked;
  index_t** colidx_blocked;
  index_t** idx_map;
  index_t** range_indices;
  float*** partial_sums;
  edata_t** edge_data_blocked;

public:
  typedef size_t iterator;
  LearningGraph(bool use_gpu) : 
        is_device(use_gpu), partitioned(false), //max_size_(0), 
        num_vertices_(0), num_edges_(0), //vertex_data_(NULL), edge_data_(NULL),
        vdata_size(0), edata_size(0),
        gpu_vsize(0), gpu_esize(0),
        rowptr_(NULL), colidx_(NULL),
        vertex_data_(NULL), edge_data_(NULL),
        d_rowptr_(NULL), d_colidx_(NULL),
        d_vertex_data_(NULL), d_edge_data_(NULL), d_trans_edge_data_(NULL) {}
  LearningGraph() : LearningGraph(false) {}
  //~LearningGraph() { dealloc(); }
  void alloc_on_device();
  void alloc_on_device(index_t n);
  void dealloc();
  size_t size() { return (size_t)num_vertices_; }
  size_t sizeEdges() { return (size_t)num_edges_; }
  bool on_device() { return is_device; }
  bool is_partitioned() { return partitioned; }
  index_t get_max_degree() { return max_degree; }
  //void init(index_t nv, index_t ne) { num_vertices_ = nv; num_edges_ = ne; }
  index_t get_degree(index_t v) { return rowptr_[v + 1] - rowptr_[v]; }
  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(num_vertices_); }
  //LearningGraph* generate_masked_graph(mask_t* masks);

  void copy_to_cpu();
  void copy_to_gpu();
  void compute_vertex_data();
  void compute_edge_data();
  //void degree_counting();
  void degree_counting() {
    index_t* degrees_ = new index_t[num_vertices_];
    #pragma omp parallel for
    for (size_t v = 0; v < num_vertices_; v++)
      degrees_[v] = rowptr_[v+1] - rowptr_[v];
    max_degree = *(std::max_element(degrees_, degrees_+num_vertices_));
    delete[] degrees_;
  }
  //void set_max_size(index_t max) { assert(max > 0); max_size_ = max; }

  // Graph reading related functions
  //void readGraph(std::string dataset, bool selfloop = false);
  //void allocateFrom(index_t nv, index_t ne);
  void fixEndEdge(index_t vid, index_t row_end) { rowptr_[vid + 1] = row_end; }
  void allocateFrom(index_t nv, index_t ne) {
    num_vertices_ = nv;
    num_edges_    = ne;
    rowptr_ = new index_t[num_vertices_ + 1];
    colidx_ = new index_t[num_edges_];
    rowptr_[0] = 0;
  }
  //void constructNodes() {}
  //void constructEdge(index_t eid, index_t dst, edata_t edata = 0);
  //void constructEdge(index_t eid, index_t dst, edata_t edata = 0) {
  void constructEdge(index_t eid, index_t dst) {
    assert(dst < num_vertices_);
    assert(eid < num_edges_);
    colidx_[eid] = dst;
    //if (edge_data_) edge_data_[eid] = edata;
  }

  index_t* row_start_host_ptr() { return &rowptr_[0]; }
  index_t*& row_host_ptr() { return *&rowptr_; }
  index_t* edge_dst_host_ptr() { return &colidx_[0]; }
  index_t*& edge_host_ptr() { return *&colidx_; }
  index_t getEdgeDstHost(index_t eid) { return colidx_[eid]; }
  index_t edge_begin_host(index_t vid) { return rowptr_[vid]; }
  index_t edge_end_host(index_t vid) { return rowptr_[vid + 1]; }

  // CSR segmenting
  void segmenting(size_t len);
  void update_feat_len(size_t len);
  index_t get_subgraph_size(int bid) { return nvs_of_subgraphs[bid]; }
  index_t get_subgraph_nedges(int bid) { return nes_of_subgraphs[bid]; }
  int get_num_subgraphs() { return num_subgraphs; }
  int get_num_ranges() { return num_ranges; }
  index_t edge_dst_blocked(int bid, index_t eid) { return colidx_blocked[bid][eid]; }
  index_t edge_begin_blocked(int bid, index_t vid) { return rowptr_blocked[bid][vid]; }
  index_t edge_end_blocked(int bid, index_t vid) { return rowptr_blocked[bid][vid+1]; }
  float* get_partial_feats(int bid, index_t vid) { return partial_sums[bid][vid]; }
  index_t get_global_vid(int bid, index_t vid) { return idx_map[bid][vid]; }
  index_t get_range_index(int bid, index_t vid) { return range_indices[bid][vid]; }

#ifndef ENABLE_GPU
  index_t getEdgeDst(index_t eid) { return colidx_[eid]; }
  index_t edge_begin(index_t vid) { return rowptr_[vid]; }
  index_t edge_end(index_t vid) { return rowptr_[vid+1]; }
  vdata_t getData(index_t vid) { return vertex_data_[vid]; }
  //index_t getDegree(index_t vid) { return degrees_[vid]; }
  index_t* row_start_ptr() { return &rowptr_[0]; }
  const index_t* row_start_ptr() const { return &rowptr_[0]; }
  index_t* edge_dst_ptr() { return &colidx_[0]; }
  const index_t* edge_dst_ptr() const { return &colidx_[0]; }
  //index_t* degrees_ptr() { return &degrees_[0]; }
  edata_t* edge_data_ptr() { return &edge_data_[0]; }
  vdata_t* vertex_data_ptr() { return &vertex_data_[0]; }
  vdata_t get_vertex_data(index_t vid) { return vertex_data_[vid]; }
  edata_t get_edge_data(index_t eid) { return edge_data_[eid]; }
#else
  CUDA_HOSTDEV index_t getEdgeDst(index_t edge) { return d_colidx_[edge]; }
  CUDA_HOSTDEV index_t edge_begin(index_t src) { return d_rowptr_[src]; }
  CUDA_HOSTDEV index_t edge_end(index_t src) { return d_rowptr_[src + 1]; }
  CUDA_HOSTDEV vdata_t getData(index_t vid) { return d_vertex_data_[vid]; }
  CUDA_HOSTDEV vdata_t get_vertex_data(index_t vid) { return d_vertex_data_[vid]; }
  CUDA_HOSTDEV edata_t get_edge_data(index_t eid) { return d_edge_data_[eid]; }
  CUDA_HOSTDEV void set_vertex_data(index_t vid, float value) { d_vertex_data_[vid] = value; }
  CUDA_HOSTDEV void set_edge_data(index_t eid, float value) { d_edge_data_[eid] = value; }
  CUDA_HOSTDEV void set_trans_edge_data(index_t eid, float value) { d_trans_edge_data_[eid] = value; }
  // CUDA_HOSTDEV index_t getDegree(index_t vid) { return d_degrees_[vid]; }
  // CUDA_HOSTDEV index_t getOutDegree(index_t vid) { return d_degrees_[vid]; }
  CUDA_HOSTDEV index_t getDegree(index_t vid) { return d_rowptr_[vid + 1] - d_rowptr_[vid]; }
  CUDA_HOSTDEV index_t getOutDegree(index_t vid) { return d_rowptr_[vid + 1] - d_rowptr_[vid]; }
  index_t* row_start_ptr() { return d_rowptr_; }
  const index_t* row_start_ptr() const { return d_rowptr_; }
  index_t* edge_dst_ptr() { return d_colidx_; }
  const index_t* edge_dst_ptr() const { return d_colidx_; }
  edata_t* edge_data_ptr() { return d_edge_data_; }
  edata_t* trans_edge_data_ptr() { return d_trans_edge_data_; }
  vdata_t* vertex_data_ptr() { return d_vertex_data_; }
  // const vdata_t *vertex_data_ptr() const { return d_vertex_data_; }
  // const edata_t *edge_data_ptr() const { return d_edge_data; }
  //index_t* degrees_ptr() { return d_degrees_; }
  void print_test();
#endif

  //void add_selfloop();
  void add_selfloop() {
    //std::cout << "Adding selfloop in the graph...\n";
    auto old_colidx_ = colidx_;
    colidx_ = new index_t[num_vertices_ + num_edges_];
    for (index_t i = 0; i < num_vertices_; i++) {
      auto start             = rowptr_[i];
      auto end               = rowptr_[i + 1];
      bool selfloop_inserted = false;
      if (start == end) {
        colidx_[start + i] = i;
        continue;
      }
      for (auto e = start; e != end; e++) {
        auto dst = old_colidx_[e];
        if (!selfloop_inserted) {
          if (i < dst) {
            selfloop_inserted  = true;
            colidx_[e + i]     = i;
            colidx_[e + i + 1] = dst;
          } else if (e + 1 == end) {
            selfloop_inserted  = true;
            colidx_[e + i + 1] = i;
            colidx_[e + i]     = dst;
          } else
            colidx_[e + i] = dst;
        } else
          colidx_[e + i + 1] = dst;
      }
    }
    for (index_t i = 0; i <= num_vertices_; i++)
      rowptr_[i] += i;
    num_edges_ += num_vertices_;
    //printf("Selfloop added: num_vertices %d num_edges %d\n", num_vertices_, num_edges_);
  }

  void print_graph() {
    std::cout << "Printing the graph: \n";
    for (index_t n = 0; n < num_vertices_; n++) {
      std::cout << "vertex " << n << ": degree = " 
                << get_degree(n) << " edgelist = [ ";
      for (auto e = edge_begin_host(n); e != edge_end_host(n); e++)
        std::cout << getEdgeDstHost(e) << " ";
      std::cout << "]\n";
    }
  }

  LearningGraph* generate_masked_graph(mask_t* masks) {
    auto n = this->size();
    LearningGraph *masked_graph = new LearningGraph();
    std::vector<uint32_t> degrees(n, 0);
    // get degrees of nodes that will be in new graph
    #pragma omp parallel for
    for (size_t src = 0; src < n; src++) {
      if (masks[src] == 1) {
        auto begin = edge_begin_host(src); 
        auto end = edge_end_host(src);
        for (auto e = begin; e != end; e++) {
          auto dst = getEdgeDstHost(e);
          if (masks[dst] == 1) degrees[src]++;
        }
      }
    }
    //auto offsets = parallel_prefix_sum(degrees);
    auto offsets = utils::prefix_sum(degrees);
    auto ne      = offsets[n];

    masked_graph->allocateFrom(n, ne);
    // same as original graph, except keep only edges involved in masks
    #pragma omp parallel for
    for (size_t src = 0; src < n; src++) {
      masked_graph->fixEndEdge(src, offsets[src + 1]);
      if (masks[src] == 1) {
        auto idx = offsets[src];
        auto begin = edge_begin_host(src); 
        auto end = edge_end_host(src);
        for (auto e = begin; e != end; e++) {
          const auto dst = getEdgeDstHost(e);
          if (masks[dst] == 1) {
            masked_graph->constructEdge(idx++, dst);
          }
        }
      }
    }
    //masked_graph->degree_counting();
    std::cout << "masked graph: num_vertices = " << masked_graph->size()
      << ", num_edges = " << masked_graph->sizeEdges() << "\n";
    return masked_graph;
  }
};

typedef LearningGraph Graph;
typedef LearningGraph GraphCPU;
typedef LearningGraph GraphGPU;
