#pragma once
#include "common.h"

using size_type = int64_t;
using bits = std::vector<bool>;

class unary_encoder {
protected:
  const size_type PRE_ENCODE_NUM = 1024 * 1024 * 16;
  int _zeta_k;
  std::vector<bits> gamma_code;
  std::vector<bits> zeta_code;
  std::vector<bits> bit_arrays; // compressed bit arrays
  std::vector<std::vector<uint32_t>> word_arrays; // compressed word arrays

public:
  explicit unary_encoder(int zeta_k) : _zeta_k(zeta_k) {}
  void pre_encoding();

  virtual size_t encode(vidType id, vidType length, vidType *in) = 0;
  virtual void print_stats() = 0;
  virtual eidType get_compressed_size(vidType i) const = 0;

  eidType get_compressed_words_size(vidType i) const { return word_arrays[i].size(); }
  eidType get_compressed_bits_size(vidType i) const { return bit_arrays[i].size(); }

  const std::vector<uint32_t> &get_compressed_words(vidType i) const { return word_arrays[i]; }
  const std::vector<std::vector<uint32_t>> &get_compressed_words() const { return word_arrays; }
  const bits &get_compressed_bits(vidType i) const { return bit_arrays[i]; }
  const std::vector<bits> &get_compressed_bits() const { return bit_arrays; }

protected:
  void print_bits(bits in);
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

