#include "cgr_compressor.hpp"

void cgr_compressor::write_cgr(const std::string &prefix) {
  FILE *of_graph = fopen((prefix + ".edge.bin").c_str(), "w");
  if (of_graph == 0) {
    std::cout << "graph file cannot create!" << std::endl;
    abort();
  }
  write_bit_array(of_graph);
  fclose(of_graph);

  std::vector<eidType> rowptr(g->V()+1);
  rowptr[0] = 0;
  for (vidType i = 0; i < g->V(); i++) {
    rowptr[i+1] = _cgr[i].bit_arr.size() + rowptr[i];
    //std::cout << "rowptr[" << i+1 << "]=" << rowptr[i+1] << "\n";
  }
  std::ofstream outfile((prefix + ".vertex.bin").c_str(), std::ios::binary);
  if (!outfile) {
    std::cout << "File not available\n";
    throw 1;
  }
  outfile.write(reinterpret_cast<const char*>(rowptr.data()), (g->V()+1)*sizeof(eidType));
  outfile.close();
}

void cgr_compressor::write_bit_array(FILE* &of) {
  std::vector<unsigned char> buf;
  unsigned char cur = 0;
  int bit_count = 0;
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
  fwrite(buf.data(), sizeof(unsigned char), buf.size(), of);
}

void cgr_compressor::encode_node(const vidType v) {
  auto &adj = this->_cgr[v];
  adj.node = v;
  adj.outd = g->get_degree(v);
  adj.itv_left.clear();
  adj.itv_len.clear();
  adj.res.clear();
  adj.bit_arr.clear();
  if (this->_res_seg_len == 0) {
    append_gamma(adj.bit_arr, adj.outd);
    if (adj.outd == 0) return;
  }
  intervalize(v);
  encode_intervals(v);
  encode_residuals(v);
}

void cgr_compressor::intervalize(const vidType v) {
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
    } else {
      for (auto i = cur_left; i < cur_right; i++) {
        adj.res.emplace_back(neighbors[i]);
      }
    }
    cur_left = cur_right;
  }
}

void cgr_compressor::encode_intervals(const vidType v) {
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
  //std::cout << "vertex " << v << " interval segment_count: " << segs.size() << "\n";
  for (size_t i = 0; i < segs.size(); i++) {
    size_type align = i + 1 == segs.size() ? 0 : this->_itv_seg_len;
    append_segment(bit_arr, segs[i].first, segs[i].second, align);
    //std::cout << "interval[" << i << "] = <" << segs[i].first << ",";
    //print_bits(segs[i].second);
    //std::cout << ">\n";
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

void cgr_compressor::encode_residuals(const vidType v) {
  auto &bit_arr = this->_cgr[v].bit_arr;
  auto &res = this->_cgr[v].res;

  typedef std::pair<size_type, bits> segment;
  std::vector<segment> segs;

  bits cur_seg;
  size_type res_cnt = 0;
  //std::cout << "res[" << v << "] = <";
  for (size_t i = 0; i < res.size(); i++) {
    //std::cout << res[i] << " ";
    size_type cur;
    if (res_cnt == 0) {
      cur = int_2_nat(res[i] - v);
    } else {
      cur = res[i] - res[i - 1] - 1;
    }
    // check if cur seg is overflowed
    if (_res_seg_len && gamma_size(res_cnt + 1) + cur_seg.size() + zeta_size(cur) > size_t(_res_seg_len)) {
      segs.emplace_back(segment(res_cnt, cur_seg));
      res_cnt = 0;
      cur = int_2_nat(res[i] - v);
      cur_seg.clear();
    }
    res_cnt++;
    append_zeta(cur_seg, cur);
  }
  //std::cout << ">\n";

  // handle last partial segment
  if (segs.empty()) {
    segs.emplace_back(segment(res_cnt, cur_seg));
  } else {
    segs.back().first += res_cnt;
    for (size_t i = res.size() - res_cnt; i < res.size(); i++) {
      append_zeta(segs.back().second, res[i] - res[i - 1] - 1);
    }
  }

  //std::cout << "vertex " << v << " residual segment_count: " << segs.size() << "\n";
  if (_res_seg_len != 0) {
    append_gamma(bit_arr, segs.size() - 1);
    for (size_t i = 0; i < segs.size(); i++) {
      size_type align = i + 1 == segs.size() ? 0 : _res_seg_len;
      append_segment(bit_arr, segs[i].first, segs[i].second, align);
      //std::cout << "residual[" << i << "] = <" << segs[i].first << ",";
      //print_bits(segs[i].second);
      //std::cout << ">\n";
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

void cgr_compressor::compress() {
  pre_encoding();
  this->_cgr.clear();
  this->_cgr.resize(g->V());
  #pragma omp parallel for
  for (vidType i = 0; i < g->V(); i++) {
    encode_node(i);
  }
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

