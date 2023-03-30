#pragma once
#include "unary_encoder.hh"
typedef std::vector<size_type> large_list;

class cgr_encoder : public unary_encoder {
  vidType total_num; // total number of arrays to compress
  bool use_interval; // use interval or not
  bool add_degree;   // attach the degree or not

  int _min_itv_len; // minimum length of an interval
  int _max_itv_len; // maximum length of an interval
  int _itv_seg_len; // number of intervals in a segment
  int _res_seg_len; // number of residuals in a segment

  int max_num_itv_per_node;            // max number of intervals in a vertex's adjlist
  int max_num_res_per_node;            // max number of residuals in a vertex's adjlist
  size_t max_num_itv_section_per_node; // max number of interval sections in a vertex's adjlist
  size_t max_num_res_section_per_node; // max number of residual sections in a vertex's adjlist
  int max_num_itv_per_section;         // max number of intervals in a section
  int max_num_res_per_section;         // max number of residuals in a section

  std::vector<large_list> interval_left; // interval start value
  std::vector<large_list> interval_len;  // interval length
  std::vector<large_list> residuals;     // residual values
  //std::vector<bits> bit_arrays; // compressed bit arrays

public:
  explicit cgr_encoder(vidType n, 
                       int zeta_k,
                       bool use_itv,
                       bool add_deg = false,
                       int min_itv_len = MIN_ITV_LEN, 
                       int itv_seg_len = INTERVAL_SEGMENT_LEN,
                       int res_seg_len = RESIDUAL_SEGMENT_LEN)
          : unary_encoder(zeta_k),
            total_num(n),
            use_interval(use_itv),
            add_degree(add_deg),
            _min_itv_len(min_itv_len),
            _max_itv_len(min_itv_len),
            _itv_seg_len(itv_seg_len),
            _res_seg_len(res_seg_len),
            max_num_itv_per_node(0),
            max_num_res_per_node(0),
            max_num_itv_section_per_node(0),
            max_num_res_section_per_node(0),
            max_num_itv_per_section(0),
            max_num_res_per_section(0) {
    interval_left.resize(total_num);
    interval_len.resize(total_num);
    residuals.resize(total_num);
    bit_arrays.resize(total_num);
    std::cout << "CGR encoder: zeta_k = " << this->_zeta_k << ", "
              << (use_interval?"interval enabled, ":"interval disabled, ")
              << (add_degree?"degree appended for all":"degree appended only for zero-residual") << " nodes\n";
  }
  size_t encode(vidType id, vidType length, vidType *in);
  void print_stats();
  eidType get_compressed_size(vidType i) const { return get_compressed_bits_size(i); }
  //eidType get_compressed_size(vidType i) const { return bit_arrays[i].size(); }
  //const bits &get_compressed_bits(vidType i) const { return bit_arrays[i]; }
  //const std::vector<bits> &get_compressed_bits() const { return bit_arrays; }

protected:
  //void encode_intervals(size_type id, size_type length, vidType *in);
  void intervalize(size_type id, size_type length, vidType *in);
  void encode_intervals(const size_type v);
  void encode_residuals(const size_type v);
  void append_segment(bits &bit_array, size_type cnt, bits &cur_seg, size_type align);
  void set_min_itv_len(int _min_itv_len) { cgr_encoder::_min_itv_len = _min_itv_len; }
  void set_itv_seg_len(int _itv_seg_len) { cgr_encoder::_itv_seg_len = _itv_seg_len; }
  void set_res_seg_len(int _res_seg_len) { cgr_encoder::_res_seg_len = _res_seg_len; }
};

