#pragma once
#include "unary_decoder.hh"
#define WORD_ALIGHED
//#define BYTE_ALIGHED
//#define USE_INTERVAL

template<typename T = vidType>
class CgrReader : public UnaryDecoder<T> {
    T id_;
  public:
    CgrReader() : UnaryDecoder<T>(NULL, 0) {}
    CgrReader(T id, T* data, OFFSET_TYPE off) :
      UnaryDecoder<T>(data, off), id_(id) { }

    void init(T id, T* data, OFFSET_TYPE off) {
      this->id_ = id;
      this->word_array = data;
      this->global_offset = off;
    }
    T get_id() { return id_; }
    T decode_segment_cnt() {
      T segment_cnt = this->decode_gamma() + 1;
      if (segment_cnt == 1 && (this->cur() & 0x80000000)) {
        this->global_offset += 1;
        segment_cnt = 0;
      }
      return segment_cnt;
    }
};

template <typename T>
class cgr_decoder {
  private:
    CgrReader<T> reader;
    T *in_ptr;
    T *out_ptr;
  public:
    cgr_decoder(T id, T *in, OFFSET_TYPE off, T* out = NULL) {
      in_ptr = in;
      out_ptr = out;
      #ifdef WORD_ALIGHED 
        reader.init(id, in, off*32); // transform word offset to bit offset
      #elif BYTE_ALIGHED 
        reader.init(id, in, off*8); // transform byte offset to bit offset
      #else
        reader.init(id, in, off);
      #endif
    }
    vidType decode();
    vidType decode_intervals();
    vidType decode_intervals(VertexList &itv_begin, VertexList &itv_end);
    vidType decode_residuals(T offset, T* out_res_ptr);
    T get_id() { return reader.get_id(); }
    OFFSET_TYPE get_offset() { return reader.get_offset(); }
};

template<typename T = vidType>
struct ResidualSegmentHelper{
  CgrReader<T> &cgrr;
  T left;
  T residual_cnt;
  bool first_res;

  ResidualSegmentHelper(CgrReader<T> &cgrr) :
    cgrr(cgrr), left(0), residual_cnt(0), first_res(true) { }

  void decode_residual_cnt() {
    this->residual_cnt = cgrr.decode_gamma();
  }
  T get_residual() {
    if (first_res) {
      left = decode_first_num();
      first_res = false;
    } else {
      left += cgrr.decode_residual_code() + 1;
    }
    residual_cnt--;
    return left;
  }
  T decode_first_num() {
    T x = cgrr.decode_residual_code();
    return (x & 1) ? cgrr.get_id() - (x >> 1) - 1 : cgrr.get_id() + (x >> 1);
  }
  T get_h() {
    return cgrr.get_h();
  }
  T get_raw_residual_value() {
    return cgrr.cur();
  }
};

template<typename T = vidType>
struct IntervalSegmentHelper {
  CgrReader<T> &cgrr;
  T left;
  T interval_cnt;
  bool first_interval;

  IntervalSegmentHelper(CgrReader<T> &cgrr) :
    cgrr(cgrr), left(0), interval_cnt(0), first_interval(true) {
  }
  void decode_interval_cnt() {
    interval_cnt = cgrr.decode_gamma();
  }
  T get_interval_left() {
    if (first_interval) {
      left = decode_first_num();
      first_interval = false;
    } else {
      left += cgrr.decode_gamma() + 1;
    }
    return left;
  }
  T get_interval_len() {
    T len = cgrr.decode_gamma() + MIN_ITV_LEN;
    left += len;
    interval_cnt--;
    return len;
  }
  T decode_first_num() {
    T x = cgrr.decode_gamma();
    return (x & 1) ? cgrr.get_id() - (x >> 1) - 1 : cgrr.get_id() + (x >> 1);
  }
};

/*
struct SeriesHelper {
  CgrReader &curp;
  vidType id_;
  vidType dout;
  vidType left;
  vidType interval_num;
  bool first_res;
  bool first_interval;

  SeriesHelper(vidType id_, CgrReader &curp, vidType dout) :
    curp(curp), id_(id_), dout(dout), left(0), 
    first_res(true), first_interval(true) {
    interval_num = dout ? curp.decode_gamma() : 0;
  }
  vidType get_interval_left() {
    if (first_interval) {
      left = curp.decode_gamma();
      left = curp.decode_first_num(id_, left);
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
      left = curp.decode_first_num(id_, left);
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
      left = curp.decode_first_num(id_, left);
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
  vidType id_;
  vidType interval_idx;
  vidType interval_num;
  vidType dout;
  vidType left;
  vidType len ;
  bool first_res;

  BaseHelper (vidType id_, CgrReader &curp, vidType dout) : 
      curp(curp), id_(id_), dout(dout) {
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
      left = curp.decode_first_num(id_, left);
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
        left = curp.decode_first_num(id_, left);
        first_res = false;
        return left;
      } else {
        left += curp.decode_residual_code() + 1;
        return left;
      }
    }
  }
};
*/

