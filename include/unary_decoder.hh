#pragma once
#include "common.h"
using OFFSET_TYPE = uint64_t;

template<typename T = vidType>
static T decode_first_num(T node, T x) {
  return (x & 1) ? node - (x >> 1) - 1 : node + (x >> 1);
}

template<typename T = vidType>
class UnaryDecoder {
  protected:
    T *word_array;
    OFFSET_TYPE global_offset;
    inline T decode_unary() {
      auto tmp = cur();
      T x = __builtin_clz(tmp);
      global_offset += x;
      return x + 1;
    }
    inline T decode_int(T len) {
      assert(len <= 32);
      T x = cur() >> (32 - len);
      global_offset += len;
      return x;
    }
    inline T get_h() {
      auto tmp = cur();
      return __builtin_clz(tmp)+1;
    }

  public:
    UnaryDecoder(T *g, OFFSET_TYPE off) : 
      word_array(g), global_offset(off) {
    }
    inline OFFSET_TYPE get_offset() { return global_offset; }
    inline void set_offset(OFFSET_TYPE off) { global_offset = off; }
    inline void inc_offset(OFFSET_TYPE off) { global_offset += off; }
    inline T decode_gamma() {
      T h = decode_unary();
      return decode_int(h) - 1;
    }
    inline T decode_zeta() {
      T h = decode_unary();
      global_offset++;
      T x = decode_int(h * ZETA_K);
      return x - 1;
    }
    inline T decode_residual_code() {
#if ZETA_K == 1
      return decode_gamma();
#else
      return decode_zeta();
#endif
    }
    inline T cur() {
      OFFSET_TYPE chunk = global_offset / 32;
      auto buf_hi = word_array[chunk];
      auto buf_lo = word_array[chunk + 1];
      T offset = global_offset % 32;
      uint64_t value = uint64_t(buf_hi) << 32 | buf_lo;
      value = (value << offset) >> 32;
      return value;
    }
};

