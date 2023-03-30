#pragma once
#include "unary_encoder.hh"

class hybrid_encoder : public unary_encoder {
private:
  vidType total_num; // total number of arrays to compress
  bool use_interval; // use interval or not
  vidType degree_threshold; // use different compressor for below and above this threshold
  std::string vbyte_scheme;
public:
  explicit hybrid_encoder(vidType n, 
                          int zeta_k = 2,
                          vidType deg = 32, 
                          bool use_itv = true,
                          std::string scheme = "streamvbyte")
          : unary_encoder(zeta_k),
            total_num(n),
            use_interval(use_itv),
            degree_threshold(deg), 
            vbyte_scheme(scheme) {
    word_arrays.resize(total_num);
    bit_arrays.resize(total_num);
  }
  vidType get_degree_threshold() { return degree_threshold; }
  void set_degree_threshold(vidType degree) { degree_threshold = degree; }
  void set_vbyte_scheme(std::string scheme) { vbyte_scheme = scheme; }

  void print_stats() {}
  size_t encode(vidType id, vidType length, vidType *in);
  eidType get_compressed_size(vidType i) const { return get_compressed_words_size(i); }
 
protected:
  size_t encode_vbyte(vidType v, vidType length, vidType *in);
  size_t encode_unary(vidType v, vidType length, vidType *in);
}; 
