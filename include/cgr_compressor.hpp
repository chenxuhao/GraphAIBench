#pragma once
#include "graph.h"

using size_type = int64_t;
using bits = std::vector<bool>;

class cgr_compressor {
  const size_type PRE_ENCODE_NUM = 1024 * 1024 * 16;

  Graph *g;
  int _zeta_k;
  int _min_itv_len;
  int _itv_seg_len;
  int _res_seg_len;

  class cgr_adjlist {
    public:
      size_type node;
      size_type outd;
      std::vector<size_type> itv_left;
      std::vector<size_type> itv_len;
      std::vector<size_type> res;
      bits bit_arr;
      cgr_adjlist() {
        node = outd = 0;
        itv_left.clear();
        itv_len.clear();
        res.clear();
        bit_arr.clear();
      }
  };

  std::vector<cgr_adjlist> _cgr;
  std::vector<bits> gamma_code;
  std::vector<bits> zeta_code;

  public:

  explicit cgr_compressor(Graph *graph, int zeta_k = 3, int min_itv_len = 4, int itv_seg_len = 0, int res_seg_len = 4 * 32)
    : g(graph), _zeta_k(zeta_k), _min_itv_len(min_itv_len),
    _itv_seg_len(itv_seg_len), _res_seg_len(res_seg_len) {}

  bool write_cgr(const std::string &dir_path) {

    bits graph;

    FILE *of_graph = fopen((dir_path + ".graph").c_str(), "w");

    if (of_graph == 0) {
      std::cout << "graph file cannot create!" << std::endl;
      abort();
    }

    this->write_bit_array(of_graph);
    fclose(of_graph);

    FILE *of_offset = fopen((dir_path + ".offset").c_str(), "w");

    if (of_offset == 0) {
      std::cout << "graph file cannot create!" << std::endl;
      abort();
    }

    size_type last_offset = 0;
    fprintf(of_offset, "%ld\n", g->V());
    for (size_type i = 0; i < g->V(); i++) {
      fprintf(of_offset, "%ld\n", this->_cgr[i].bit_arr.size() + last_offset);
      last_offset += this->_cgr[i].bit_arr.size();
    }
    fclose(of_offset);
    return true;
  }

  void write_bit_array(FILE* &of) {
    std::vector<unsigned char> buf;
    unsigned char cur = 0;
    int bit_count = 0;
    for (size_type i = 0; i < g->V(); i++) {
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

  void encode_node(const size_type node) {

    auto &adj = this->_cgr[node];

    adj.node = node;
    adj.outd = g->get_degree(node);
    adj.itv_left.clear();
    adj.itv_len.clear();
    adj.res.clear();
    adj.bit_arr.clear();

    if (this->_res_seg_len == 0) {
      append_gamma(adj.bit_arr, adj.outd);
      if (adj.outd == 0) return;
    }

    intervalize(node);
    encode_intervals(node);
    encode_residuals(node);
  }

  void intervalize(const size_type node) {
    size_type cur_left = 0, cur_right = 0;
    auto deg = g->get_degree(node);
    auto neighbors = g->N(node);
    auto &adj = this->_cgr[node];

    while (cur_left < deg) {
      cur_right = cur_left + 1;
      while (cur_right < deg && neighbors[cur_right - 1] + 1 == neighbors[cur_right]) cur_right++;
      size_type cur_len = cur_right - cur_left;
      if ((cur_len >= this->_min_itv_len) && (this->_min_itv_len != 0)) {
        adj.itv_left.emplace_back(neighbors[cur_left]);
        adj.itv_len.emplace_back(cur_len);
      } else {
        for (size_type i = cur_left; i < cur_right; i++) {
          adj.res.emplace_back(neighbors[i]);
        }
      }
      cur_left = cur_right;
    }
  }

  void encode_intervals(const size_type node) {
    auto &bit_arr = this->_cgr[node].bit_arr;
    auto &itv_left = this->_cgr[node].itv_left;
    auto &itv_len = this->_cgr[node].itv_len;

    typedef std::pair<size_type, bits> segment;
    std::vector<segment> segs;

    bits cur_seg;
    size_type itv_cnt = 0;
    for (size_type i = 0; i < itv_left.size(); i++) {
      size_type cur_left = 0;
      if (itv_cnt == 0) {
        cur_left = int_2_nat(itv_left[i] - node);
      } else {
        cur_left = itv_left[i] - itv_left[i - 1] - itv_len[i - 1] - 1;
      }
      size_type cur_len = itv_len[i] - this->_min_itv_len;

      // check if cur seg is overflowed
      if (this->_itv_seg_len &&
          gamma_size(itv_cnt + 1) + cur_seg.size() + gamma_size(cur_left) + gamma_size(cur_len) >
          this->_itv_seg_len) {
        segs.emplace_back(segment(itv_cnt, cur_seg));
        itv_cnt = 0;
        cur_left = int_2_nat(itv_left[i] - node);
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
      for (size_type i = itv_left.size() - itv_cnt; i < itv_left.size(); i++) {
        append_gamma(segs.back().second, itv_left[i] - itv_left[i - 1] - itv_len[i - 1] - 1);
        append_gamma(segs.back().second, itv_len[i] - this->_min_itv_len);
      }
    }

    if (this->_itv_seg_len != 0) append_gamma(bit_arr, segs.size() - 1);
    for (size_type i = 0; i < segs.size(); i++) {
      size_type align = i + 1 == segs.size() ? 0 : this->_itv_seg_len;
      append_segment(bit_arr, segs[i].first, segs[i].second, align);
    }
  }

  void encode_residuals(const size_type node) {
    auto &bit_arr = this->_cgr[node].bit_arr;
    auto &res = this->_cgr[node].res;

    typedef std::pair<size_type, bits> segment;
    std::vector<segment> segs;

    bits cur_seg;
    size_type res_cnt = 0;
    for (size_type i = 0; i < res.size(); i++) {

      size_type cur;
      if (res_cnt == 0) {
        cur = int_2_nat(res[i] - node);
      } else {
        cur = res[i] - res[i - 1] - 1;
      }

      // check if cur seg is overflowed
      if (this->_res_seg_len && gamma_size(res_cnt + 1) + cur_seg.size() + zeta_size(cur) > this->_res_seg_len) {
        segs.emplace_back(segment(res_cnt, cur_seg));
        res_cnt = 0;
        cur = int_2_nat(res[i] - node);
        cur_seg.clear();
      }

      res_cnt++;
      append_zeta(cur_seg, cur);
    }

    // handle last partial segment
    if (segs.empty()) {
      segs.emplace_back(segment(res_cnt, cur_seg));
    } else {
      segs.back().first += res_cnt;
      for (size_type i = res.size() - res_cnt; i < res.size(); i++) {
        append_zeta(segs.back().second, res[i] - res[i - 1] - 1);
      }
    }

    if (this->_res_seg_len != 0) {
      append_gamma(bit_arr, segs.size() - 1);
      for (size_type i = 0; i < segs.size(); i++) {
        size_type align = i + 1 == segs.size() ? 0 : this->_res_seg_len;
        append_segment(bit_arr, segs[i].first, segs[i].second, align);
      }
    } else {
      bit_arr.insert(bit_arr.end(), cur_seg.begin(), cur_seg.end());
    }
  }

  void append_segment(bits &bit_array, size_type cnt, bits &cur_seg, size_type align) {
    bits buf;
    append_gamma(buf, cnt);
    buf.insert(buf.end(), cur_seg.begin(), cur_seg.end());

    assert(align == 0 or buf.size() <= align);
    while (buf.size() < align) buf.emplace_back(false);

    bit_array.insert(bit_array.end(), buf.begin(), buf.end());
  }

  void append_gamma(bits &bit_array, size_type x) {
    if (x < this->PRE_ENCODE_NUM) {
      bit_array.insert(bit_array.end(), this->gamma_code[x].begin(), this->gamma_code[x].end());
    } else {
      encode_gamma(bit_array, x);
    }
  }

  void append_zeta(bits &bit_array, size_type x) {
    if (x < this->PRE_ENCODE_NUM) {
      bit_array.insert(bit_array.end(), this->zeta_code[x].begin(), this->zeta_code[x].end());
    } else {
      encode_zeta(bit_array, x);
    }
  }

  size_type int_2_nat(size_type x) {
    return x >= 0L ? x << 1 : -((x << 1) + 1L);
  }

  size_type gamma_size(size_type x) {
    if (x < this->PRE_ENCODE_NUM) return this->gamma_code[x].size();
    x++;
    assert(x >= 0);
    int len = this->get_significent_bit(x);
    return 2 * len + 1;
  }

  size_type zeta_size(size_type x) {
    if (x < this->PRE_ENCODE_NUM) return this->zeta_code[x].size();
    x++;
    assert(x >= 0);
    int len = this->get_significent_bit(x);
    int h = len / this->_zeta_k;
    return (h + 1) * (this->_zeta_k + 1);
  }


  void compress() {
    pre_encoding();

    this->_cgr.clear();
    this->_cgr.resize(g->V());

#pragma omp parallel for
    for (size_type i = 0; i < g->V(); i++) {
      encode_node(i);
    }
  }

  void pre_encoding() {
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

  void encode_gamma(bits &bit_array, size_type x) {
    x++;
    assert(x >= 0);
    int len = this->get_significent_bit(x);
    this->encode(bit_array, 1, len + 1);
    this->encode(bit_array, x, len);
  }

  void encode_zeta(bits &bit_array, size_type x) {
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

  void encode(bits &bit_array, size_type x, int len) {
    for (int i = len - 1; i >= 0; i--) {
      bit_array.emplace_back((x >> i) & 1L);
    }
  }

  int get_significent_bit(size_type x) {
    assert(x > 0);
    int ret = 0;
    while (x > 1) x >>= 1, ret++;
    return ret;
  }

  void set_zeta_k(int _zeta_k) {
    cgr_compressor::_zeta_k = _zeta_k;
  }

  void set_min_itv_len(int _min_itv_len) {
    cgr_compressor::_min_itv_len = _min_itv_len;
  }

  void set_itv_seg_len(int _itv_seg_len) {
    cgr_compressor::_itv_seg_len = _itv_seg_len;
  }

  void set_res_seg_len(int _res_seg_len) {
    cgr_compressor::_res_seg_len = _res_seg_len;
  }
};

