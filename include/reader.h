#pragma once
#include "lgraph.h"

class Reader {
private:
  std::string dataset_str;
  void progressPrint(unsigned maxi, unsigned i);

public:
  Reader() : dataset_str("") {}
  Reader(std::string dataset) : dataset_str(dataset) {}
  void init(std::string dataset) { dataset_str = dataset; }
  size_t read_labels(std::vector<label_t>& labels, bool is_single_class = true);
  size_t read_features(std::vector<float>& feats, std::string filetype = "bin");
  size_t read_masks(std::string mask_type, size_t n, size_t& begin, size_t& end, mask_t* masks);
  void readGraphFromGRFile(LearningGraph* g);
};

