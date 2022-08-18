#pragma once
#include "graph.h"

using size_type = int64_t;
using bits = std::vector<bool>;

class cgr_compressor {
  const size_type PRE_ENCODE_NUM = 1024 * 1024 * 16;

  Graph *g;
  int _zeta_k;
  int _min_itv_len; // minimum length of an interval
  int _max_itv_len; // maximum length of an interval
  int _itv_seg_len; // number of intervals in a segment
  int _res_seg_len; // number of residuals in a segment
  int max_num_itv_per_node; // max number of intervals in a node's adjlist
  int max_num_res_per_node; // max number of residuals in a node's adjlist

  class cgr_adjlist {
    public:
      size_type node;
      size_type outd; // out degree
      std::vector<size_type> itv_left;
      std::vector<size_type> itv_len;
      std::vector<size_type> res;
      bits bit_arr;
      cgr_adjlist() {
        node = outd = 0;
        itv_left.clear();
        itv_len.clear();
        res.clear();
        bit_arr.clear();
      }
  };

  std::vector<cgr_adjlist> _cgr;
  std::vector<bits> gamma_code;
  std::vector<bits> zeta_code;

public:
  explicit cgr_compressor(Graph *graph, int zeta_k = 3, int min_itv_len = 4, 
                          int itv_seg_len = 0, int res_seg_len = 4 * 32)
          : g(graph), _zeta_k(zeta_k), _min_itv_len(min_itv_len), _max_itv_len(min_itv_len),
            _itv_seg_len(itv_seg_len), _res_seg_len(res_seg_len),
            max_num_itv_per_node(0), max_num_res_per_node(0) {}
  void compress();
  void write_cgr(const std::string &dir_path);
  void write_bit_array(FILE* &of);

protected:
  void print_bits(bits in);
  void encode_node(const size_type v);
  void intervalize(const size_type v);
  void encode_intervals(const size_type v);
  void encode_residuals(const size_type v);
  void append_segment(bits &bit_array, size_type cnt, bits &cur_seg, size_type align);
  void append_gamma(bits &bit_array, size_type x);
  void append_zeta(bits &bit_array, size_type x);
  size_type int_2_nat(size_type x);
  size_type gamma_size(size_type x);
  size_type zeta_size(size_type x);
  void pre_encoding();
  void encode_gamma(bits &bit_array, size_type x);
  void encode_zeta(bits &bit_array, size_type x);
  void encode(bits &bit_array, size_type x, int len);
  int get_significent_bit(size_type x);
  void set_zeta_k(int _zeta_k) { cgr_compressor::_zeta_k = _zeta_k; }
  void set_min_itv_len(int _min_itv_len) { cgr_compressor::_min_itv_len = _min_itv_len; }
  void set_itv_seg_len(int _itv_seg_len) { cgr_compressor::_itv_seg_len = _itv_seg_len; }
  void set_res_seg_len(int _res_seg_len) { cgr_compressor::_res_seg_len = _res_seg_len; }
};

