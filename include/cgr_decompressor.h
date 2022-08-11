#pragma once
#include "common.h"

class CgrReader {
  public:
    vidType node;
    vidType *graph;
    eidType global_offset;

    CgrReader(vidType v, vidType *g, eidType off) :
      node(v), graph(g), global_offset(off) { }

    void init(vidType v, vidType *graph, eidType global_offset) {
      this->node = v;
      this->graph = graph;
      this->global_offset = global_offset;
    }
    static SIZE_TYPE decode_first_num(vidType node, SIZE_TYPE x) {
      return (x & 1) ? node - (x >> 1) - 1 : node + (x >> 1);
    }
    vidType cur() {
      eidType chunk = global_offset / 32;
      auto buf_hi = graph[chunk];
      auto buf_lo = graph[chunk + 1];
      vidType offset = global_offset % 32;
      int64_t value = int64_t(buf_hi) << 32 | buf_lo;
      return (value << offset) >> 32;
      //return __funnelshift_l(buf_lo, buf_hi, offset);
    }
    SIZE_TYPE decode_unary() {
      auto tmp = cur();
      SIZE_TYPE x = __builtin_clz(tmp);
      global_offset += x;
      return x + 1;
    }
    SIZE_TYPE decode_int(SIZE_TYPE len) {
      SIZE_TYPE x = cur() >> (32 - len);
      global_offset += len;
      return x;
    }
    SIZE_TYPE decode_gamma() {
      SIZE_TYPE h = decode_unary();
      return this->decode_int(h) - 1;
    }
#if ZETA_K != 1
    SIZE_TYPE decode_zeta() {
      SIZE_TYPE h = decode_unary();
      global_offset++;
      SIZE_TYPE x = decode_int(h * ZETA_K);
      return x - 1;
    }
#endif
    SIZE_TYPE decode_residual_code() {
#if ZETA_K == 1
      return decode_gamma();
#else
      return decode_zeta();
#endif
    }
    SIZE_TYPE decode_segment_cnt() {
      SIZE_TYPE segment_cnt = decode_gamma() + 1;
      if (segment_cnt == 1 && (cur() & 0x80000000)) {
        global_offset += 1;
        segment_cnt = 0;
      }
      return segment_cnt;
    }
};

struct ResidualSegmentHelper{
  CgrReader &cgrr;
  SIZE_TYPE left;
  SIZE_TYPE residual_cnt;
  bool first_res;

  ResidualSegmentHelper(SIZE_TYPE node, CgrReader &cgrr) :
    cgrr(cgrr), left(0), residual_cnt(0), first_res(true) { }

  void decode_residual_cnt() {
    this->residual_cnt = cgrr.decode_gamma();
  }
  SIZE_TYPE get_residual() {
    if (first_res) {
      left = decode_first_num();
      first_res = false;
    } else {
      left += cgrr.decode_residual_code() + 1;
    }
    residual_cnt--;
    return left;
  }
  SIZE_TYPE decode_first_num() {
    SIZE_TYPE x = cgrr.decode_residual_code();
    return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
  }

};

struct IntervalSegmentHelper {
  CgrReader &cgrr;
  SIZE_TYPE left;
  SIZE_TYPE interval_cnt;
  bool first_interval;

  IntervalSegmentHelper(SIZE_TYPE node, CgrReader &cgrr) :
    cgrr(cgrr), left(0), interval_cnt(0), first_interval(true) {
  }
  void decode_interval_cnt() {
    interval_cnt = cgrr.decode_gamma();
  }
  SIZE_TYPE get_interval_left() {
    if (first_interval) {
      left = decode_first_num();
      first_interval = false;
    } else {
      left += cgrr.decode_gamma() + 1;
    }
    return left;
  }
  SIZE_TYPE get_interval_len() {
    SIZE_TYPE len = cgrr.decode_gamma() + MIN_ITV_LEN;
    left += len;
    interval_cnt--;
    return len;
  }
  SIZE_TYPE decode_first_num() {
    SIZE_TYPE x = cgrr.decode_gamma();
    return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
  }
};

struct SeriesHelper {
  CgrReader &curp;
  SIZE_TYPE node;
  SIZE_TYPE dout;
  SIZE_TYPE left;
  SIZE_TYPE interval_num;
  bool first_res;
  bool first_interval;

  SeriesHelper(SIZE_TYPE node, CgrReader &curp, SIZE_TYPE dout) :
    curp(curp), node(node), dout(dout), left(0), 
    first_res(true), first_interval(true) {
    interval_num = dout ? curp.decode_gamma() : 0;
  }
  SIZE_TYPE get_interval_left() {
    if (first_interval) {
      left = curp.decode_gamma();
      left = curp.decode_first_num(node, left);
      first_interval = false;
    } else {
      left += curp.decode_gamma() + 1;
    }
    return left;
  }
  SIZE_TYPE get_interval_len() {
    SIZE_TYPE len = curp.decode_gamma() + MIN_ITV_LEN;
    dout -= len;
    left += len;
    return len;
  }
  SIZE_TYPE get_residual() {
    if (first_res) {
      left = curp.decode_residual_code();
      left = curp.decode_first_num(node, left);
      first_res = false;
    } else {
      left += curp.decode_residual_code() + 1;
    }
    dout--;
    return left;
  }
  SIZE_TYPE calc_residual(SIZE_TYPE x) {
    if (first_res) {
      left = x;
      left = curp.decode_first_num(node, left);
      first_res = false;
    } else {
      left += x + 1;
    }
    dout--;
    return left;
  }
};

struct BaseHelper {
  CgrReader &curp;
  SIZE_TYPE node;
  SIZE_TYPE interval_idx;
  SIZE_TYPE interval_num;
  SIZE_TYPE dout;
  SIZE_TYPE left;
  SIZE_TYPE len ;
  bool first_res;

  BaseHelper (SIZE_TYPE node, CgrReader &curp, SIZE_TYPE dout) : 
      curp(curp), node(node), dout(dout) {
    if (dout) {
      interval_num = curp.decode_gamma();
      interval_idx = 0;
      len = 0;
      refresh_interval();
      first_res = true;
    }
  }
  void refresh_interval() {
    if (interval_idx >= interval_num) return;
    if (len) return;
    if (interval_idx == 0) {
      left = curp.decode_gamma();
      left = curp.decode_first_num(node, left);
    } else {
      left += curp.decode_gamma() + 1;
    }
    len = curp.decode_gamma() + MIN_ITV_LEN;
    interval_idx++;
  }
  SIZE_TYPE fetch_next() {
    dout--;
    if (len) {
      // interval
      SIZE_TYPE cur = left;
      left++;
      len--;
      refresh_interval();
      return cur;
    } else {
      // residual
      if (first_res) {
        left = curp.decode_residual_code();
        left = curp.decode_first_num(node, left);
        first_res = false;
        return left;
      } else {
        left += curp.decode_residual_code() + 1;
        return left;
      }
    }
  }
};

