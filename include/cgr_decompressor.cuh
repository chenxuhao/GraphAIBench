#pragma once
#include "common.h"

class CgrReaderGPU {
  public:
    eidType global_offset;
    vidType *graph;
    vidType node;

    __device__ void init(vidType node, vidType *graph, eidType global_offset) {
      this->node = node;
      this->graph = graph;
      this->global_offset = global_offset;
    }

    static __device__ vidType decode_first_num(vidType node, vidType x) {
      return (x & 1) ? node - (x >> 1) - 1 : node + (x >> 1);
    }

    __device__ vidType cur() {
      eidType chunk = global_offset / 32;
      vidType buf_hi = graph[chunk];
      vidType buf_lo = graph[chunk + 1];
      vidType offset = global_offset % 32;
      return __funnelshift_l(buf_lo, buf_hi, offset);
    }

    __device__ vidType decode_unary() {
      vidType tmp = cur();
      vidType x = __clz(tmp);
      global_offset += x;
      return x + 1;
    }

    __device__ vidType decode_int(vidType len) {
      vidType x = cur() >> (32 - len);
      global_offset += len;
      return x;
    }

    __device__ vidType decode_gamma() {
      vidType h = decode_unary();
      return this->decode_int(h) - 1;
    }

#if ZETA_K != 1
    __device__ vidType decode_zeta() {
      vidType h = decode_unary();
      global_offset++;
      vidType x = decode_int(h * ZETA_K);
      return x - 1;
    }
#endif

    __device__ vidType decode_residual_code() {
#if ZETA_K == 1
      return decode_gamma();
#else
      return decode_zeta();
#endif
    }

    __device__ vidType decode_segment_cnt() {
      vidType segment_cnt = node == SIZE_NONE ? 0 : decode_gamma() + 1;
      if (segment_cnt == 1 && (cur() & 0x80000000)) {
        global_offset += 1;
        segment_cnt = 0;
      }
      return segment_cnt;
    }
};

struct ResidualSegmentHelperGPU {
  vidType residual_cnt;
  vidType left;
  bool first_res;
  CgrReader &cgrr;

  __device__ ResidualSegmentHelperGPU(vidType node, CgrReader &cgrr) :
    cgrr(cgrr), first_res(true), left(0), residual_cnt(0) {
    }

  __device__ void decode_residual_cnt() {
    this->residual_cnt = cgrr.node == SIZE_NONE ? 0 : cgrr.decode_gamma();
  }

  __device__ vidType get_residual() {
    if (first_res) {
      left = decode_first_num();
      first_res = false;
    } else {
      left += cgrr.decode_residual_code() + 1;
    }
    residual_cnt--;
    return left;
  }

  __device__ vidType decode_first_num() {
    vidType x = cgrr.decode_residual_code();
    return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
  }

};

struct IntervalSegmentHelperGPU {
  vidType interval_cnt;
  vidType left;
  bool first_interval;
  CgrReader &cgrr;

  __device__ IntervalSegmentHelperGPU(vidType node, CgrReader &cgrr) :
    cgrr(cgrr), first_interval(true), left(0), interval_cnt(0) {
    }

  __device__ void decode_interval_cnt() {
    interval_cnt = cgrr.node == SIZE_NONE ? 0 : cgrr.decode_gamma();
  }

  __device__ vidType get_interval_left() {
    if (first_interval) {
      left = decode_first_num();
      first_interval = false;
    } else {
      left += cgrr.decode_gamma() + 1;
    }
    return left;
  }

  __device__ vidType get_interval_len() {
    vidType len = cgrr.decode_gamma() + MIN_ITV_LEN;
    left += len;
    interval_cnt--;
    return len;
  }

  __device__ vidType decode_first_num() {
    vidType x = cgrr.decode_gamma();
    return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
  }
};

struct SeriesHelperGPU {
  vidType interval_num;
  vidType node;
  vidType dout;
  vidType left;
  bool first_res;
  bool first_interval;
  CgrReader &curp;

  __device__ SeriesHelperGPU(vidType node, CgrReader &curp, vidType dout) :
    node(node), curp(curp), dout(dout), first_res(true), first_interval(true) {
      interval_num = dout ? curp.decode_gamma() : 0;
    }

  __device__ vidType get_interval_left() {
    if (first_interval) {
      left = curp.decode_gamma();
      left = curp.decode_first_num(node, left);
      first_interval = false;
    } else {
      left += curp.decode_gamma() + 1;
    }
    return left;
  }

  __device__ vidType get_interval_len() {
    vidType len = curp.decode_gamma() + MIN_ITV_LEN;
    dout -= len;
    left += len;
    return len;
  }

  __device__ vidType get_residual() {
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

  __device__ vidType calc_residual(vidType x) {
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

struct BaseHelperGPU {
  vidType interval_idx;
  vidType interval_num;
  vidType node;
  vidType dout;
  vidType left;
  vidType len ;
  bool first_res;
  CgrReader &curp;

  __device__ BaseHelperGPU (vidType node, CgrReader &curp, vidType dout) :
    node(node), curp(curp), dout(dout) {
      if (dout) {
        interval_num = curp.decode_gamma();
        interval_idx = 0;
        len = 0;
        refresh_interval();
        first_res = true;
      }
    }

  __device__ void refresh_interval() {
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

  __device__ vidType fetch_next() {
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

