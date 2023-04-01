#include "VertexSet.h"
#include "cgr_decoder.hh"

template <typename T>
vidType cgr_decoder<T>::decode_intervals() {
  vidType num = 0;
  // handle segmented intervals
  auto segment_cnt = reader.decode_segment_cnt();
  //std::cout << "decoding vertex " << get_id() << " intervals: segment_cnt=" << segment_cnt << "\n";
  // for each segment
  auto interval_offset = reader.get_offset();
  for (T i = 0; i < segment_cnt; i++) {
    CgrReader<T> cgrr(reader.get_id(), in_ptr, interval_offset);
    IntervalSegmentHelper isHelper(cgrr);
    isHelper.decode_interval_cnt();
    auto num_intervals = isHelper.interval_cnt;
    // for each interval in the segment
    for (T j = 0; j < num_intervals; j++) {
      auto left = isHelper.get_interval_left();
      auto len = isHelper.get_interval_len();
      //assert(left < n_vertices);
      //assert(len < max_degree);
      for (T k = 0; k < len; k++) {
        out_ptr[num++] = left+k;
      }
    }
    interval_offset += INTERVAL_SEGMENT_LEN;
    if (i == segment_cnt-1) {// last segment
      reader.set_offset(cgrr.get_offset());
    } else
      reader.inc_offset(INTERVAL_SEGMENT_LEN);
  }
  return num;
}

template <typename T>
vidType cgr_decoder<T>::decode_intervals(VertexList &itv_begin, VertexList &itv_end) {
  //std::cout << "decoding vertex " << get_id() << " intervals: ";
  vidType num = 0;
  // handle segmented intervals
  auto segment_cnt = reader.decode_segment_cnt();
  //std::cout << "segment_cnt=" << segment_cnt << "\n";
  // for each segment
  auto interval_offset = reader.get_offset();
  for (T i = 0; i < segment_cnt; i++) {
    CgrReader<T> cgrr(reader.get_id(), in_ptr, interval_offset);
    IntervalSegmentHelper isHelper(cgrr);
    isHelper.decode_interval_cnt();
    auto num_intervals = isHelper.interval_cnt;
    //std::cout << "num_intervals=" << num_intervals << "\n";
    // for each interval in the segment
    for (T j = 0; j < num_intervals; j++) {
      auto left = isHelper.get_interval_left();
      auto len = isHelper.get_interval_len();
      //assert(left < n_vertices);
      //assert(len < max_degree);
      itv_begin.push_back(left);
      itv_end.push_back(left+len);
      //std::cout << "left=" << left << " len=" << len << "\n";
      num += len;
    }
    interval_offset += INTERVAL_SEGMENT_LEN;
    if (i == segment_cnt-1) // last segment
      reader.set_offset(cgrr.get_offset());
    else
      reader.inc_offset(INTERVAL_SEGMENT_LEN);
  }
  return num;
}

template <typename T>
vidType cgr_decoder<T>::decode_residuals(T offset, T* out_res_ptr) {
  vidType num = offset;
  // handle segmented residuals
  auto segment_cnt = reader.decode_segment_cnt();
  //std::cout << "decoding vertex " << get_id() << " residuals, segment_cnt=" << segment_cnt << " offset=" << num << "\n";
  auto residual_offset = reader.get_offset();
  for (T i = 0; i < segment_cnt; i++) {
    CgrReader<T> cgrr(reader.get_id(), in_ptr, residual_offset);
    ResidualSegmentHelper rsHelper(cgrr);
    rsHelper.decode_residual_cnt();
    auto num_res = rsHelper.residual_cnt;
    //std::cout << "out_res_ptr=" << out_res_ptr << " num_res=" << num_res << " adj_list: [ ";
    // for each residual in the segment
    for (T j = 0; j < num_res; j++) {
      auto residual = rsHelper.get_residual();
      //assert(residual < n_vertices);
      //std::cout << residual << " ";
      out_res_ptr[num++] = residual;
    }
    //std::cout << "]\n";
    residual_offset += RESIDUAL_SEGMENT_LEN;
    reader.inc_offset(RESIDUAL_SEGMENT_LEN);
  }
  return num;
}

template <typename T>
vidType cgr_decoder<T>::decode() {
  //std::cout << "decode() vertex " << get_id() << " in_ptr=" << in_ptr << " out_ptr=" << out_ptr << " offset=" << get_offset() << "\n";
  vidType num = 0;
#if USE_INTERVAL
  num = decode_intervals();
#endif
  num = decode_residuals(num, out_ptr);
  return num;
}

template class cgr_decoder<vidType>;
