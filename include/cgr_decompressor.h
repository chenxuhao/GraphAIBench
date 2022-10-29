#pragma once
#include "common.h"
using OFFSET_TYPE = uint64_t;

class CgrReader {
  public:
    vidType node;
    vidType *graph;
    OFFSET_TYPE global_offset;

    CgrReader(vidType v, vidType *g, OFFSET_TYPE off) :
      node(v), graph(g), global_offset(off) { }

    void init(vidType v, vidType *graph, OFFSET_TYPE global_offset) {
      this->node = v;
      this->graph = graph;
      this->global_offset = global_offset;
    }
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
      //if (value < 0) printf("node=%d\n", node);
      //assert(value >= 0);
      return value;
      //return __funnelshift_l(buf_lo, buf_hi, offset);
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
#if ZETA_K != 1
    vidType decode_zeta() {
      vidType h = decode_unary();
      global_offset++;
      vidType x = decode_int(h * ZETA_K);
      return x - 1;
    }
#endif
    vidType decode_residual_code() {
#if ZETA_K == 1
      return decode_gamma();
#else
      return decode_zeta();
#endif
    }
    vidType decode_segment_cnt() {
      vidType segment_cnt = decode_gamma() + 1;
      if (segment_cnt == 1 && (cur() & 0x80000000)) {
        global_offset += 1;
        segment_cnt = 0;
      }
      return segment_cnt;
    }
};

struct ResidualSegmentHelper{
  CgrReader &cgrr;
  vidType left;
  vidType residual_cnt;
  bool first_res;

  ResidualSegmentHelper(vidType node, CgrReader &cgrr) :
    cgrr(cgrr), left(0), residual_cnt(0), first_res(true) { }

  void decode_residual_cnt() {
    this->residual_cnt = cgrr.decode_gamma();
  }
  vidType get_residual() {
    if (first_res) {
      left = decode_first_num();
      first_res = false;
    } else {
      left += cgrr.decode_residual_code() + 1;
    }
    residual_cnt--;
    return left;
  }
  vidType decode_first_num() {
    vidType x = cgrr.decode_residual_code();
    return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
  }
  vidType get_h() {
    return cgrr.get_h();
  }
  vidType get_raw_residual_value() {
    return cgrr.cur();
  }
};

struct IntervalSegmentHelper {
  CgrReader &cgrr;
  vidType left;
  vidType interval_cnt;
  bool first_interval;

  IntervalSegmentHelper(vidType node, CgrReader &cgrr) :
    cgrr(cgrr), left(0), interval_cnt(0), first_interval(true) {
  }
  void decode_interval_cnt() {
    interval_cnt = cgrr.decode_gamma();
  }
  vidType get_interval_left() {
    if (first_interval) {
      left = decode_first_num();
      first_interval = false;
    } else {
      left += cgrr.decode_gamma() + 1;
    }
    return left;
  }
  vidType get_interval_len() {
    vidType len = cgrr.decode_gamma() + MIN_ITV_LEN;
    left += len;
    interval_cnt--;
    return len;
  }
  vidType decode_first_num() {
    vidType x = cgrr.decode_gamma();
    return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
  }
};

struct SeriesHelper {
  CgrReader &curp;
  vidType node;
  vidType dout;
  vidType left;
  vidType interval_num;
  bool first_res;
  bool first_interval;

  SeriesHelper(vidType node, CgrReader &curp, vidType dout) :
    curp(curp), node(node), dout(dout), left(0), 
    first_res(true), first_interval(true) {
    interval_num = dout ? curp.decode_gamma() : 0;
  }
  vidType get_interval_left() {
    if (first_interval) {
      left = curp.decode_gamma();
      left = curp.decode_first_num(node, left);
      first_interval = false;
    } else {
      left += curp.decode_gamma() + 1;
    }
    return left;
  }
  vidType get_interval_len() {
    vidType len = curp.decode_gamma() + MIN_ITV_LEN;
    dout -= len;
    left += len;
    return len;
  }
  vidType get_residual() {
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
  vidType calc_residual(vidType x) {
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
  vidType node;
  vidType interval_idx;
  vidType interval_num;
  vidType dout;
  vidType left;
  vidType len ;
  bool first_res;

  BaseHelper (vidType node, CgrReader &curp, vidType dout) : 
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
  vidType fetch_next() {
    dout--;
    if (len) {
      // interval
      vidType cur = left;
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

