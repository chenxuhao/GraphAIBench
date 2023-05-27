#pragma once
#include "lgraph.h"

class Reader {
private:
  std::string dataset_str;
  void progressPrint(unsigned maxi, unsigned i);
  std::string name_;          // name of the graph
  std::string inputfile_path; // file path of the graph

  index_t feat_len;       // vertex feature vector length: '0' means no features
  int num_vertex_classes; // number of distinct vertex labels: '0' means no vertex labels
  int num_edge_classes;   // number of distinct edge labels: '0' means no edge labels
  index_t num_vertices_;
  index_t num_edges_;
  int32_t *edges;    // column indices of CSR format
  index_t *vertices; // row pointers of CSR format
  vlabel_t *vlabels; // vertex labels
  int train_begin, train_end, train_count;
  int val_begin, val_end, val_count;
  int test_begin, test_end, test_count;

public:
  Reader() : dataset_str("") {}
  Reader(std::string dataset) : dataset_str(dataset) {}
  void init(std::string dataset) { dataset_str = dataset; }
  

  size_t csgr_read_labels(std::vector<label_t> &labels, bool is_single_class = true);
  size_t csgr_read_features(std::vector<float> &feats, std::string filetype = "bin");
  size_t csgr_read_masks(std::string mask_type, size_t n, size_t &begin, size_t &end, mask_t *masks);
  void csgr_read_graph(LearningGraph *g);

  size_t bin_read_features(std::vector<float> &feats);
  size_t bin_read_masks(std::string mask_type, size_t n, size_t &begin, size_t &end, mask_t *masks);
  template <typename T>
  void bin_read_file(std::string fname, T *&pointer, size_t length);
  void bin_read_graph(LearningGraph *g);
  int bin_read_vlabels(std::vector<label_t> &labels, bool is_single_class = true);
};

