#pragma once
#include "graph.h"

using size_type = int64_t;
using bits = std::vector<bool>;
typedef OutOfCoreGraph GraphTy;

class unary_compressor {
protected:
  const size_type PRE_ENCODE_NUM = 1024 * 1024 * 16;
  GraphTy *g;
  int _zeta_k;
  std::vector<bits> gamma_code;
  std::vector<bits> zeta_code;

public:
  explicit unary_compressor(GraphTy *graph, int zeta_k)
          : g(graph), _zeta_k(zeta_k) {}
  void compress();
  void write_graph(const std::string &dir_path);
  void write_bit_array(FILE* &of);

protected:
  void print_bits(bits in);
  //void encode_vertex(const size_type v);
  void pre_encoding();
  void append_gamma(bits &bit_array, size_type x);
  void append_zeta(bits &bit_array, size_type x);
  size_type int_2_nat(size_type x);
  size_type gamma_size(size_type x);
  size_type zeta_size(size_type x);
  void encode_gamma(bits &bit_array, size_type x);
  void encode_zeta(bits &bit_array, size_type x);
  void encode(bits &bit_array, size_type x, int len);
  int get_significent_bit(size_type x);
  void set_zeta_k(int zeta_k) { _zeta_k = zeta_k; }
}; 
