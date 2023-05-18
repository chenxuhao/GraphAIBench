#pragma once
#include "unary_decoder.hh"
#define ALIGHEMENT 1

template <typename T = vidType>
class cgr_decoder : public UnaryDecoder<T> {
  private:
    T id_;
    T *out_ptr;
    int res_seg_len; // number of residuals in a segment
  public:
    cgr_decoder() : UnaryDecoder<T>(NULL, 0) {}
    cgr_decoder(T id, T *in, OFFSET_TYPE off, T* out = NULL, int rlen = 256) :
      UnaryDecoder<T>(in, off*ALIGHEMENT),
      id_(id), out_ptr(out), res_seg_len(rlen) { }
    vidType decode();
    vidType decode_intervals();
    vidType decode_intervals(VertexList &itv_begin, VertexList &itv_end);
    vidType decode_residuals(T offset, T* out_res_ptr);
    inline T get_id() { return id_; }
  private:
    inline T decode_segment_cnt() {
      T segment_cnt = this->decode_gamma() + 1;
      if (segment_cnt == 1 && (this->cur() & 0x80000000)) {
        this->global_offset += 1;
        segment_cnt = 0;
      }
      return segment_cnt;
    }
};

