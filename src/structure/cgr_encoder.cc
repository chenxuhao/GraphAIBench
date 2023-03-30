#include "cgr_encoder.hh"

void cgr_encoder::print_stats() {
  std::cout << "max_num_itv_per_node = " << max_num_itv_per_node << "\n"
            << "max_num_res_per_node = " << max_num_res_per_node << "\n"
            << "max_num_itv_section_per_node = " << max_num_itv_section_per_node << "\n"
            << "max_num_res_section_per_node = " << max_num_res_section_per_node << "\n"
            << "max_num_itv_per_section = " << max_num_itv_per_section << "\n"
            << "max_num_res_per_section = " << max_num_res_per_section << "\n"
            << "max_itv_len = " << _max_itv_len << "\n";
}

// encode an integer array "*in" with "length" elements using CGR format
size_t cgr_encoder::encode(vidType id, vidType length, vidType *in) {
  interval_left[id].clear();
  interval_len[id].clear();
  residuals[id].clear();
  bit_arrays[id].clear();
  //std::cout << "encoding vid=" << id << ", degree=" << length << "\n";
  //std::cout << "add_degree=" << add_degree << ", _res_seg_len=" << _res_seg_len << "\n";
  if (add_degree || _res_seg_len == 0) {
    //std::cout << "appending degree=" << length << "\n";
    append_gamma(bit_arrays[id], length);
    if (length == 0) return 0;
  }
  if (use_interval) {
    intervalize(id, length, in);
    //encode_intervals(id, length, in);
    encode_intervals(id);
  } else {
    residuals[id].assign(in, in+length);
  }
  encode_residuals(id);

  interval_left[id].clear();
  interval_len[id].clear();
  residuals[id].clear();
  return (bit_arrays[id].size() - 1)/8 + 1; // number of bits --> number of bytes
}

//void cgr_encoder::encode_intervals(size_type id, size_type length, vidType *in) {
void cgr_encoder::intervalize(size_type id, size_type length, vidType *in) {
  size_type cur_left = 0, cur_right = 0;
  auto &itv_left = interval_left[id];
  auto &itv_len = interval_len[id];
  size_type deg = length;

  while (cur_left < deg) {
    cur_right = cur_left + 1;
    while (cur_right < deg && in[cur_right - 1] + 1 == in[cur_right]) cur_right++;
    auto cur_len = cur_right - cur_left;
    if ((cur_len >= this->_min_itv_len) && (this->_min_itv_len != 0)) {
      itv_left.emplace_back(in[cur_left]);
      itv_len.emplace_back(cur_len);
      if (cur_len > _max_itv_len) _max_itv_len = cur_len;
    } else {
      for (auto i = cur_left; i < cur_right; i++) {
        residuals[id].emplace_back(in[i]);
      }
    }
    cur_left = cur_right;
  }
  auto num_intervals = itv_left.size();
  auto num_residuals = residuals[id].size();
  if (size_t(max_num_itv_per_node) < num_intervals)
    max_num_itv_per_node = num_intervals;
  if (size_t(max_num_res_per_node) < num_residuals)
    max_num_res_per_node = num_residuals;
}

void cgr_encoder::encode_intervals(const size_type v) {
  auto &bit_arr = bit_arrays[v];
  auto &itv_left = interval_left[v];
  auto &itv_len = interval_len[v];

  typedef std::pair<size_type, bits> segment;
  std::vector<segment> segs;

  bits cur_seg;
  size_type itv_cnt = 0;
  for (size_t i = 0; i < itv_left.size(); i++) {
    size_type cur_left = 0;
    if (itv_cnt == 0) {
      cur_left = int_2_nat(itv_left[i] - v);
    } else {
      cur_left = itv_left[i] - itv_left[i - 1] - itv_len[i - 1] - 1;
    }
    size_type cur_len = itv_len[i] - this->_min_itv_len;

    // check if cur seg is overflowed
    if (_itv_seg_len &&
        gamma_size(itv_cnt + 1) + cur_seg.size() + gamma_size(cur_left) + gamma_size(cur_len) >
        size_t(_itv_seg_len)) {
      segs.emplace_back(segment(itv_cnt, cur_seg));
      if (max_num_itv_per_section < itv_cnt) max_num_itv_per_section = itv_cnt;
      itv_cnt = 0;
      cur_left = int_2_nat(itv_left[i] - v);
      cur_seg.clear();
    }
    itv_cnt++;
    append_gamma(cur_seg, cur_left);
    append_gamma(cur_seg, cur_len);
  }

  // handle last paritial segment
  if (segs.empty()) {
    segs.emplace_back(segment(itv_cnt, cur_seg));
    if (max_num_itv_per_section < itv_cnt) max_num_itv_per_section = itv_cnt;
  } else {
    segs.back().first += itv_cnt;
    for (size_t i = itv_left.size() - itv_cnt; i < itv_left.size(); i++) {
      append_gamma(segs.back().second, itv_left[i] - itv_left[i - 1] - itv_len[i - 1] - 1);
      append_gamma(segs.back().second, itv_len[i] - this->_min_itv_len);
    }
    if (max_num_itv_per_section < itv_cnt) max_num_itv_per_section = itv_cnt;
  }

  if (max_num_itv_section_per_node < segs.size()) max_num_itv_section_per_node = segs.size();
  if (this->_itv_seg_len != 0) append_gamma(bit_arr, segs.size() - 1);
  for (size_t i = 0; i < segs.size(); i++) {
    size_type align = i + 1 == segs.size() ? 0 : this->_itv_seg_len;
    append_segment(bit_arr, segs[i].first, segs[i].second, align);
  }
}

void cgr_encoder::encode_residuals(const size_type v) {
  auto &bit_arr = bit_arrays[v];
  auto &res = residuals[v];

  typedef std::pair<size_type, bits> segment;
  std::vector<segment> segs;

  bits cur_seg;
  size_type res_cnt = 0;
  int segment_id = 0;
  for (size_t i = 0; i < res.size(); i++) {
    size_type cur;
    if (res_cnt == 0) {
      cur = int_2_nat(res[i] - v);
    } else {
      assert(i>=1);
      cur = res[i] - res[i - 1] - 1;
    }
    // check if cur seg is overflowed
    if (_res_seg_len && gamma_size(res_cnt + 1) + cur_seg.size() + zeta_size(cur) > size_t(_res_seg_len)) {
      segs.emplace_back(segment(res_cnt, cur_seg));
      if (max_num_res_per_section < res_cnt)
        max_num_res_per_section = res_cnt;
      res_cnt = 0;
      cur = int_2_nat(res[i] - v);
      cur_seg.clear();
      segment_id ++;
    }
    res_cnt++;
    append_zeta(cur_seg, cur);
  }

  // handle last partial segment
  if (segs.empty()) {
    segs.emplace_back(segment(res_cnt, cur_seg));
  } else {
    segs.back().first += res_cnt;
    for (size_t i = res.size() - res_cnt; i < res.size(); i++) {
      assert(i>=1);
      append_zeta(segs.back().second, res[i] - res[i - 1] - 1);
    }
  }

  if (max_num_res_section_per_node < segs.size()) max_num_res_section_per_node = segs.size();
  if (_res_seg_len != 0) {
    append_gamma(bit_arr, segs.size() - 1);
    for (size_t i = 0; i < segs.size(); i++) {
      size_type align = i + 1 == segs.size() ? 0 : _res_seg_len;
      append_segment(bit_arr, segs[i].first, segs[i].second, align);
    }
  } else {
    bit_arr.insert(bit_arr.end(), cur_seg.begin(), cur_seg.end());
  }
}

void cgr_encoder::append_segment(bits &bit_array, size_type cnt, bits &cur_seg, size_type align) {
  bits buf;
  append_gamma(buf, cnt);
  buf.insert(buf.end(), cur_seg.begin(), cur_seg.end());
  assert(align == 0 or buf.size() <= size_t(align));
  while (buf.size() < size_t(align)) buf.emplace_back(false);
  bit_array.insert(bit_array.end(), buf.begin(), buf.end());
}

