#include "VertexSet.h"
#include "cgr_decoder.hh"
//#define USE_INTERVAL

template <typename T>
vidType cgr_decoder<T>::decode_intervals() {
  vidType num = 0;
  // decode the number of segments
  auto segment_cnt = decode_segment_cnt();
  // for each segment
  for (T i = 0; i < segment_cnt; i++) {
    auto off = this->get_offset();
    // decode the number of intervals in the segment
    auto num_intervals = this->decode_gamma();
    // decode the first interval
    T x = this->decode_gamma();
    auto left = (x & 1) ? get_id() - (x >> 1) - 1 : get_id() + (x >> 1);
    T len = this->decode_gamma() + MIN_ITV_LEN;
    for (T k = 0; k < len; k++) out_ptr[num++] = left+k;
    left += len;
    // for each interval in the segment
    for (T j = 1; j < num_intervals; j++) {
      left += this->decode_gamma() + 1;
      len = this->decode_gamma() + MIN_ITV_LEN;
      for (T k = 0; k < len; k++) out_ptr[num++] = left+k;
      left += len;
    }
    if (i != segment_cnt-1) // if not the last segment
      this->set_offset(off+INTERVAL_SEGMENT_LEN);
  }
  return num;
}

template <typename T>
vidType cgr_decoder<T>::decode_intervals(VertexList &itv_begin, VertexList &itv_end) {
  vidType num = 0;
  auto segment_cnt = decode_segment_cnt();
  for (T i = 0; i < segment_cnt; i++) {
    auto off = this->get_offset();
    auto num_intervals = this->decode_gamma();
    T x = this->decode_gamma();
    auto left = (x & 1) ? get_id() - (x >> 1) - 1 : get_id() + (x >> 1);
    T len = this->decode_gamma() + MIN_ITV_LEN;
    itv_begin.push_back(left);
    itv_end.push_back(left+len);
    left += len;
    for (T j = 1; j < num_intervals; j++) {
      left += this->decode_gamma() + 1;
      len = this->decode_gamma() + MIN_ITV_LEN;
      itv_begin.push_back(left);
      itv_end.push_back(left+len);
      left += len;
      num += len;
    }
    if (i != segment_cnt-1) // last segment
      this->set_offset(off+INTERVAL_SEGMENT_LEN);
  }
  return num;
}

//#define res_seg_len 256
template <typename T>
vidType cgr_decoder<T>::decode_residuals(T offset, T* out_res_ptr) {
  vidType num = offset;
  // decode the number of segments
  auto segment_cnt = decode_segment_cnt();
  //printf("decoding residuls for vertex %d, segment_cnt %d\n", id_, segment_cnt);
  // for each segment
  for (T i = 0; i < segment_cnt; i++) {
    auto off = this->get_offset();
    // decode the number of residuals in the segment
    auto num_res = this->decode_gamma();
    // decode the first residual in the segment
    T x = this->decode_residual_code();
    T value = (x & 1) ? id_ - (x >> 1) - 1 : id_ + (x >> 1);
    out_res_ptr[num++] = value;
    //printf("adj[0]=%d ", value);
    // decode the rest of residuals in the segment
    for (T j = 1; j < num_res; j++) {
      value += this->decode_residual_code() + 1;
      out_res_ptr[num++] = value;
      //printf("adj[%d]=%d ", j, value);
    }
    //printf("\n");
    this->set_offset(off+res_seg_len);
  }
  return num;
}

template <typename T>
vidType cgr_decoder<T>::decode() {
  //printf("decoding vertex %d\n", get_id());
  vidType num = 0;
#ifdef USE_INTERVAL
  num = decode_intervals();
#endif
  num = decode_residuals(num, out_ptr);
  return num;
}

template class cgr_decoder<vidType>;
