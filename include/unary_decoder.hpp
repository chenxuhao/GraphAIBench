#pragma once
#include "common.h"
using OFFSET_TYPE = uint64_t;

class UnaryDecoder {
  protected:
    vidType *graph;
    OFFSET_TYPE global_offset;

  public:
    UnaryDecoder(vidType *g, OFFSET_TYPE off) : 
      graph(g), global_offset(off) {
    }
    OFFSET_TYPE get_offset() { return global_offset; }
    void set_offset(OFFSET_TYPE off) { global_offset = off; }
    void inc_offset(OFFSET_TYPE off) { global_offset += off; }
    static vidType decode_first_num(vidType node, vidType x) {
      return (x & 1) ? node - (x >> 1) - 1 : node + (x >> 1);
    }
    vidType cur() {
      OFFSET_TYPE chunk = global_offset / 32;
      auto buf_hi = graph[chunk];
      auto buf_lo = graph[chunk + 1];
      vidType offset = global_offset % 32;
      uint64_t value = uint64_t(buf_hi) << 32 | buf_lo;
      value = (value << offset) >> 32;
      return value;
    }
    vidType decode_unary() {
      auto tmp = cur();
      vidType x = __builtin_clz(tmp);
      global_offset += x;
      return x + 1;
    }
    vidType decode_int(vidType len) {
      assert(len <= 32);
      vidType x = cur() >> (32 - len);
      global_offset += len;
      return x;
    }
    vidType get_h() {
      auto tmp = cur();
      return __builtin_clz(tmp)+1;
    }
    vidType decode_gamma() {
      vidType h = decode_unary();
      return this->decode_int(h) - 1;
    }
    vidType decode_zeta() {
      vidType h = decode_unary();
      global_offset++;
      vidType x = decode_int(h * ZETA_K);
      return x - 1;
    }
};

