#include "cgr_decompressor.hpp"

vidType cgr_decompressor::decode_segment_cnt(eidType &offset) {
  auto segment_cnt = node == SIZE_NONE ? 0 : decode_gamma() + 1;
  if (segment_cnt == 1 && (cur() & 0x80000000)) {
    offset += 1;
    segment_cnt = 0;
  }
  return segment_cnt;
}

