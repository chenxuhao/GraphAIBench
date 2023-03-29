#include "unary_encoder.hh"

void unary_encoder::print_bits(bits in) {
  std::cout << "0b";
  for (size_t i = 0; i < in.size(); i++) {
    if (in[i]) {
      std::cout << 1;
    } else {
      std::cout << 0;
    }
  }
}

void unary_encoder::append_gamma(bits &bit_array, size_type x) {
  //std::cout << "appending gmma code for value " << x << "\n";
  if (x < this->PRE_ENCODE_NUM) {
    bit_array.insert(bit_array.end(), this->gamma_code[x].begin(), this->gamma_code[x].end());
  } else {
    encode_gamma(bit_array, x);
  }
}

void unary_encoder::append_zeta(bits &bit_array, size_type x) {
  //std::cout << "appending zeta code for value " << x << "\n";
  if (x < this->PRE_ENCODE_NUM) {
    bit_array.insert(bit_array.end(), this->zeta_code[x].begin(), this->zeta_code[x].end());
  } else {
    encode_zeta(bit_array, x);
  }
}

size_type unary_encoder::int_2_nat(size_type x) {
  return x >= 0L ? x << 1 : -((x << 1) + 1L);
}

size_type unary_encoder::gamma_size(size_type x) {
  if (x < this->PRE_ENCODE_NUM) return this->gamma_code[x].size();
  x++;
  assert(x >= 0);
  int len = this->get_significent_bit(x);
  return 2 * len + 1;
}

size_type unary_encoder::zeta_size(size_type x) {
  if (x < this->PRE_ENCODE_NUM) return this->zeta_code[x].size();
  x++;
  assert(x >= 0);
  int len = this->get_significent_bit(x);
  int h = len / this->_zeta_k;
  return (h + 1) * (this->_zeta_k + 1);
}

void unary_encoder::pre_encoding() {
  auto num = PRE_ENCODE_NUM;
  this->gamma_code.clear();
  this->gamma_code.resize(num);
  this->zeta_code.clear();
  this->zeta_code.resize(num);
  #pragma omp parallel for
  for (size_type i = 0; i < num; i++) {
    // pre-encode gamma
    encode_gamma(this->gamma_code[i], i);
    // pre-encode zeta
    if (this->_zeta_k == 1) {
      this->zeta_code[i] = this->gamma_code[i];
    } else {
      encode_zeta(this->zeta_code[i], i);
    }
  }
}

void unary_encoder::encode_gamma(bits &bit_array, size_type x) {
  x++;
  assert(x >= 0);
  int len = this->get_significent_bit(x);
  this->encode(bit_array, 1, len + 1);
  this->encode(bit_array, x, len);
}

void unary_encoder::encode_zeta(bits &bit_array, size_type x) {
  if (this->_zeta_k == 1) {
    encode_gamma(bit_array, x);
  } else {
    x++;
    assert(x >= 0);
    int len = this->get_significent_bit(x);
    int h = len / this->_zeta_k;

    // NOTE: Encoding should not be longer than 32-bit
    // so when _zeta_k = 3, vid should not exceed 31-bit,
    // otherwise, the significent bit > 30, and h >= 10
    // h+1 >= 11, ((h+1)*_zeta_k) >= 33
    // For gsh-2015 and bigger graphs, use _zeta_k=2 or _zeta_k=1
    assert((h+1)*this->_zeta_k <= 32);

    this->encode(bit_array, 1, h + 1);
    this->encode(bit_array, x, (h + 1) * this->_zeta_k);
  }
}

void unary_encoder::encode(bits &bit_array, size_type x, int len) {
  for (int i = len - 1; i >= 0; i--) {
    bit_array.emplace_back((x >> i) & 1L);
  }
}

int unary_encoder::get_significent_bit(size_type x) {
  assert(x > 0);
  int ret = 0;
  while (x > 1) x >>= 1, ret++;
  return ret;
}

