#include "cgr_compressor.hpp"

void cgr_compressor::write_cgr(const std::string &prefix) {
  std::cout << "writing the compressed data to disk\n";
  FILE *of_graph = fopen((prefix + ".edge.bin").c_str(), "w");
  if (of_graph == 0) {
    std::cout << "graph file cannot create!" << std::endl;
    abort();
  }
  write_bit_array(of_graph);
  fclose(of_graph);

  std::cout << "Computing the row pointers\n";
  Timer t;
  t.Start();
  std::vector<eidType> rowptr(g->V()+1);
#if 0
  std::vector<vidType> degrees(g->V());
  #pragma omp parallel for
  for (vidType i = 0; i < g->V(); i++)
    degrees[i] = _cgr[i].bit_arr.size();
  parallel_prefix_sum<vidType,eidType>(degrees, rowptr.data());
#else
  rowptr[0] = 0;
  for (vidType i = 0; i < g->V(); i++)
    rowptr[i+1] = _cgr[i].bit_arr.size() + rowptr[i];
#endif
  t.Stop();
  std::cout << "Computing row pointers time: " << t.Seconds() << "\n";

  std::cout << "Writing the row pointers to disk\n";
  t.Start();
  std::ofstream outfile((prefix + ".vertex.bin").c_str(), std::ios::binary);
  if (!outfile) {
    std::cout << "File not available\n";
    throw 1;
  }
  outfile.write(reinterpret_cast<const char*>(rowptr.data()), (g->V()+1)*sizeof(eidType));
  outfile.close();
  t.Stop();
  std::cout << "Writing row pointers time: " << t.Seconds() << "\n";
}

void cgr_compressor::write_bit_array(FILE* &of) {
  std::cout << "writing the bit array\n";
  Timer t;
  t.Start();
  std::vector<unsigned char> buf;
  unsigned char cur = 0;
  int bit_count = 0;
#if 1
  for (vidType i = 0; i < g->V(); i++) {
    for (auto bit : this->_cgr[i].bit_arr) {
      cur <<= 1;
      if (bit) cur++;
      bit_count++;
      if (bit_count == 8) {
        buf.emplace_back(cur);
        cur = 0;
        bit_count = 0;
      }
    }
  }
  if (bit_count) {
    while (bit_count < 8) cur <<= 1, bit_count++;
    buf.emplace_back(cur);
  }
#else
#endif
  fwrite(buf.data(), sizeof(unsigned char), buf.size(), of);
  t.Stop();
  std::cout << "Writing bit array time: " << t.Seconds() << "\n";
}

void cgr_compressor::encode_node(const size_type v, bool use_interval, bool add_degree) {
  auto &adj = this->_cgr[v];
  adj.node = v;
  adj.outd = g->get_degree(v);
  adj.itv_left.clear();
  adj.itv_len.clear();
  adj.res.clear();
  adj.bit_arr.clear();
  if (add_degree || this->_res_seg_len == 0) {
    append_gamma(adj.bit_arr, adj.outd);
    if (adj.outd == 0) return;
  }
  if (use_interval) {
    intervalize(v);
    encode_intervals(v);
  } else {
    adj.res.assign(g->N(v).begin(), g->N(v).end());
  }
  encode_residuals(v);
}

void cgr_compressor::intervalize(const size_type v) {
  size_type cur_left = 0, cur_right = 0;
  auto deg = g->get_degree(v);
  auto neighbors = g->N(v);
  auto &adj = this->_cgr[v];

  while (cur_left < deg) {
    cur_right = cur_left + 1;
    while (cur_right < deg && neighbors[cur_right - 1] + 1 == neighbors[cur_right]) cur_right++;
    auto cur_len = cur_right - cur_left;
    if ((cur_len >= this->_min_itv_len) && (this->_min_itv_len != 0)) {
      adj.itv_left.emplace_back(neighbors[cur_left]);
      adj.itv_len.emplace_back(cur_len);
      if (cur_len > _max_itv_len) _max_itv_len = cur_len;
    } else {
      for (auto i = cur_left; i < cur_right; i++) {
        adj.res.emplace_back(neighbors[i]);
      }
    }
    cur_left = cur_right;
  }
  auto num_intervals = adj.itv_left.size();
  auto num_residuals = adj.res.size();
  if (size_t(max_num_itv_per_node) < num_intervals)
    max_num_itv_per_node = num_intervals;
  if (size_t(max_num_res_per_node) < num_residuals)
    max_num_res_per_node = num_residuals;
}

void cgr_compressor::encode_intervals(const size_type v) {
  auto &bit_arr = this->_cgr[v].bit_arr;
  auto &itv_left = this->_cgr[v].itv_left;
  auto &itv_len = this->_cgr[v].itv_len;

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
  } else {
    segs.back().first += itv_cnt;
    for (size_t i = itv_left.size() - itv_cnt; i < itv_left.size(); i++) {
      append_gamma(segs.back().second, itv_left[i] - itv_left[i - 1] - itv_len[i - 1] - 1);
      append_gamma(segs.back().second, itv_len[i] - this->_min_itv_len);
    }
  }

  if (this->_itv_seg_len != 0) append_gamma(bit_arr, segs.size() - 1);
  for (size_t i = 0; i < segs.size(); i++) {
    size_type align = i + 1 == segs.size() ? 0 : this->_itv_seg_len;
    append_segment(bit_arr, segs[i].first, segs[i].second, align);
  }
}

void cgr_compressor::print_bits(bits in) {
  std::cout << "0b";
  for (size_t i = 0; i < in.size(); i++) {
    if (in[i]) {
      std::cout << 1;
    } else {
      std::cout << 0;
    }
  }
}

void cgr_compressor::encode_residuals(const size_type v) {
  auto &bit_arr = this->_cgr[v].bit_arr;
  auto &res = this->_cgr[v].res;

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
      res_cnt = 0;
      cur = int_2_nat(res[i] - v);
      cur_seg.clear();
      segment_id ++;
    }
    res_cnt++;
    //auto cur_seg_start = cur_seg.size();
    append_zeta(cur_seg, cur);
    //auto cur_seg_end = cur_seg.size();
    /*
    if (v == 446 && segment_id == 2) {
      std::cout << "v " << v << " neighbor[" << i << "]=" << res[i] << "\n"; 
      std::cout << "v " << v << " neighbor[" << i << "] encoded as: " << cur;
      if (res_cnt == 1) std::cout << " the first in the segment; ";
      //std::cout << " supposed to be " << int_2_nat(res[i] - v);
      auto len = get_significent_bit(cur+1);
      int h = len / this->_zeta_k;
      std::cout << " len: " << len << " h: " << h << "\n";

      std::cout << "[" << cur_seg_end-cur_seg_start << "-bit]";
      for (auto j = cur_seg_start; j < cur_seg_end; j++) {
        if (cur_seg[j]) {
          std::cout << 1;
        } else {
          std::cout << 0;
        }
      }
      std::cout << "\n";
    }
    //*/
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

  if (_res_seg_len != 0) {
    //if (v == 446) std::cout << "v " << v << " number of residual segments: " << segs.size() << "\n";
    append_gamma(bit_arr, segs.size() - 1);
    for (size_t i = 0; i < segs.size(); i++) {
      size_type align = i + 1 == segs.size() ? 0 : _res_seg_len;
      append_segment(bit_arr, segs[i].first, segs[i].second, align);
    }
  } else {
    bit_arr.insert(bit_arr.end(), cur_seg.begin(), cur_seg.end());
  }
}

void cgr_compressor::append_segment(bits &bit_array, size_type cnt, bits &cur_seg, size_type align) {
  bits buf;
  append_gamma(buf, cnt);
  buf.insert(buf.end(), cur_seg.begin(), cur_seg.end());
  assert(align == 0 or buf.size() <= size_t(align));
  while (buf.size() < size_t(align)) buf.emplace_back(false);
  bit_array.insert(bit_array.end(), buf.begin(), buf.end());
}

void cgr_compressor::append_gamma(bits &bit_array, size_type x) {
  if (x < this->PRE_ENCODE_NUM) {
    bit_array.insert(bit_array.end(), this->gamma_code[x].begin(), this->gamma_code[x].end());
  } else {
    encode_gamma(bit_array, x);
  }
}

void cgr_compressor::append_zeta(bits &bit_array, size_type x) {
  if (x < this->PRE_ENCODE_NUM) {
    bit_array.insert(bit_array.end(), this->zeta_code[x].begin(), this->zeta_code[x].end());
  } else {
    encode_zeta(bit_array, x);
  }
}

size_type cgr_compressor::int_2_nat(size_type x) {
  return x >= 0L ? x << 1 : -((x << 1) + 1L);
}

size_type cgr_compressor::gamma_size(size_type x) {
  if (x < this->PRE_ENCODE_NUM) return this->gamma_code[x].size();
  x++;
  assert(x >= 0);
  int len = this->get_significent_bit(x);
  return 2 * len + 1;
}

size_type cgr_compressor::zeta_size(size_type x) {
  if (x < this->PRE_ENCODE_NUM) return this->zeta_code[x].size();
  x++;
  assert(x >= 0);
  int len = this->get_significent_bit(x);
  int h = len / this->_zeta_k;
  return (h + 1) * (this->_zeta_k + 1);
}

void cgr_compressor::compress(bool use_interval, bool add_degree) {
  std::cout << "Start compressing: zeta_k = " << this->_zeta_k << ", "
    << (use_interval?"interval enabled, ":"interval disabled, ")
    << (add_degree?"degree appended for all":"degree appended only for zero-residual") << " nodes\n";
  Timer t;
  t.Start();
  max_num_res_per_node = 0;
  pre_encoding();
  t.Stop();
  std::cout << "Pre-encoding time: " << t.Seconds() << "\n";

  this->_cgr.clear();
  this->_cgr.resize(g->V());
  t.Start();
  #pragma omp parallel for
  for (vidType i = 0; i < g->V(); i++) {
    encode_node(i, use_interval, add_degree);
  }
  std::cout << "max_num_itv_per_node = " << max_num_itv_per_node 
            << " max_num_res_per_node = " << max_num_res_per_node
            << " max_itv_len = " << _max_itv_len << "\n";
  t.Stop();
  std::cout << "Encoding time: " << t.Seconds() << "\n";
}

void cgr_compressor::pre_encoding() {
  this->gamma_code.clear();
  this->gamma_code.resize(this->PRE_ENCODE_NUM);
  this->zeta_code.clear();
  this->zeta_code.resize(this->PRE_ENCODE_NUM);
  #pragma omp parallel for
  for (size_type i = 0; i < this->PRE_ENCODE_NUM; i++) {
    // pre-encode gamma
    encode_gamma(this->gamma_code[i], i);
    // pre-encode zeta
    if (this->_zeta_k == 1) {
      this->zeta_code[i] = this->gamma_code[i];
    } else {
      encode_zeta(this->zeta_code[i], i);
    }
  }
}

void cgr_compressor::encode_gamma(bits &bit_array, size_type x) {
  x++;
  assert(x >= 0);
  int len = this->get_significent_bit(x);
  this->encode(bit_array, 1, len + 1);
  this->encode(bit_array, x, len);
}

void cgr_compressor::encode_zeta(bits &bit_array, size_type x) {
  if (this->_zeta_k == 1) {
    encode_gamma(bit_array, x);
  } else {
    x++;
    assert(x >= 0);
    int len = this->get_significent_bit(x);
    int h = len / this->_zeta_k;

    // NOTE: Encoding should not be longer than 32-bit
    // so when _zeta_k = 3, vid should not exceed 31-bit,
    // otherwise, the significent bit > 30, and h >= 10
    // h+1 >= 11, ((h+1)*_zeta_k) >= 33
    // For gsh-2015 and bigger graphs, use _zeta_k=2 or _zeta_k=1
    assert((h+1)*this->_zeta_k <= 32);

    this->encode(bit_array, 1, h + 1);
    this->encode(bit_array, x, (h + 1) * this->_zeta_k);
  }
}

void cgr_compressor::encode(bits &bit_array, size_type x, int len) {
  for (int i = len - 1; i >= 0; i--) {
    bit_array.emplace_back((x >> i) & 1L);
  }
}

int cgr_compressor::get_significent_bit(size_type x) {
  assert(x > 0);
  int ret = 0;
  while (x > 1) x >>= 1, ret++;
  return ret;
}

