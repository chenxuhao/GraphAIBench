#pragma once
#include "defines.h"
#include "common.h"
#include "timer.h"
#include "custom_alloc.h"
constexpr vidType VID_MIN = 0;
constexpr vidType VID_MAX = std::numeric_limits<vidType>::max();

inline vidType bs(vidType* ptr, int set_size, vidType o){
  int idx_l = -1;
  int idx_r = set_size;
  //guarantees in this area is that idx_l is before where we would put o 
  while (idx_r-idx_l > 1){
    auto idx_t = (idx_l+idx_r)/2;
    if (ptr[idx_t] < o) idx_l = idx_t;
    else idx_r = idx_t;
  }
  return idx_l+1;
}

class VertexSet {
private: // memory managed regions for per-thread intermediates
  vidType *ptr;
  vidType set_size, vid;
  const bool pooled;
  static thread_local std::vector<vidType*> buffers_exist, buffers_avail;

public:
  static void release_buffers();
  static vidType MAX_DEGREE;

  VertexSet() : VertexSet(vidType(-1)) {}
  VertexSet(vidType v) : set_size(0), vid(v), pooled(true) {
    if(buffers_avail.size() == 0) { 
      vidType *p = custom_alloc_local<vidType>(MAX_DEGREE);
      buffers_exist.push_back(p);
      buffers_avail.push_back(p);
    }
    ptr = buffers_avail.back();
    buffers_avail.pop_back();
  }
  VertexSet(vidType *p, vidType s, vidType id) : 
    ptr(p), set_size(s), vid(id), pooled(false) {}
  VertexSet(const VertexSet&)=delete;
  VertexSet& operator=(const VertexSet&)=delete;
  VertexSet(VertexSet&&)=default;
  VertexSet& operator=(VertexSet&&)=default;

  void duplicate(const VertexSet &other) {
    ptr = other.ptr;
    set_size = other.set_size;
    vid = other.vid;
  }

  ~VertexSet() {
    if(pooled) {
      buffers_avail.push_back(ptr);
    }
  }

  vidType size() const { return set_size; }
  void adjust_size(vidType s) { set_size = s; }
  vidType get_vid() const { return vid; }
  vidType *data() const { return ptr; }
  const vidType *begin() const { return ptr; }
  const vidType *end() const { return ptr+set_size; }
  void add(vidType v) { ptr[set_size++] = v; }
  void clear() { set_size = 0; }
  vidType& operator[](size_t i) { return ptr[i]; }
  const vidType& operator[](size_t i) const { return ptr[i]; }
  void sort() { std::sort(ptr, ptr+set_size); }

  VertexSet operator &(const VertexSet &other) const {
    VertexSet out;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) out.ptr[out.set_size++] = left;
    }
    return out;
  }
  uint32_t get_intersect_num(const VertexSet &other) const {
    uint32_t num = 0;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) num++;
    }
    return num;
  }

  void print() const {
    std::copy(ptr, ptr+set_size, std::ostream_iterator<vidType>(std::cout, " "));
  }

  vidType difference_buf(vidType *outBuf, const VertexSet &other) const;

  VertexSet operator -(const VertexSet &other) const {
    VertexSet out;
    out.set_size = difference_buf(out.ptr, other); 
    return out;
  }

  VertexSet& difference(VertexSet& dst, const VertexSet &other) const {
    dst.set_size = difference_buf(dst.ptr, other);
    return dst;
  }

  VertexSet intersect(const VertexSet &other, vidType upper) const {
    VertexSet out;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) out.ptr[out.set_size++] = left;
    }
    return out;
  }

  vidType intersect_ns(const VertexSet &other, vidType upper) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) idx_out++;
    }
    return idx_out;
  }

  vidType intersect_ns_except(const VertexSet &other, vidType upper, vidType ancestor) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right && left != ancestor) idx_out++;
    }
    return idx_out;
  }

  vidType intersect_except(const VertexSet &other, vidType ancestor) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right && left != ancestor) idx_out++;
    }
    return idx_out;
  }

  vidType intersect_except(const VertexSet &other, vidType ancestorA, vidType ancestorB) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right && left != ancestorA && left != ancestorB) idx_out++;
    }
    return idx_out;
  }

  //outBuf may be the same as this->ptr
  vidType difference_buf(vidType *outBuf, const VertexSet &other, vidType upper) const;

  VertexSet difference(const VertexSet &other, vidType upper) const {
    VertexSet out;
    out.set_size = difference_buf(out.ptr, other, upper);
    return out;
  }

  VertexSet& difference(VertexSet& dst, const VertexSet &other, vidType upper) const {
    dst.set_size = difference_buf(dst.ptr, other, upper);
    return dst;
  }

  vidType difference_ns(const VertexSet &other, vidType upper) const;

  VertexSet bounded(vidType up) const {
    if (set_size > 64) {
      vidType idx_l = vidType(-1);
      vidType idx_r = set_size;
      while (idx_r-idx_l > 1) {
        vidType idx_t = (idx_l+idx_r)/2;
        if (ptr[idx_t] < up) idx_l = idx_t;
        else idx_r = idx_t;
      }
      return VertexSet(ptr, idx_l+1, vid);
    } else {
      vidType idx_l = 0;
      while (idx_l < set_size && ptr[idx_l] < up) ++idx_l;
      return VertexSet(ptr, idx_l, vid);
    }
  }
};

inline VertexSet difference_set(const VertexSet& a, const VertexSet& b) {
  return a-b;
}

inline VertexSet& difference_set(VertexSet& dst, const VertexSet& a, const VertexSet& b) {
  return a.difference(dst, b);
}

inline VertexSet difference_set(const VertexSet& a, const VertexSet& b, vidType up) {
  return a.difference(b,up);
}

inline VertexSet& difference_set(VertexSet& dst, const VertexSet& a, const VertexSet& b, vidType up) {
  return a.difference(dst, b,up);
}

inline uint64_t difference_num(const VertexSet& a, const VertexSet& b) {
  return (a-b).size();
}

inline uint64_t difference_num(const VertexSet& a, const VertexSet& b, vidType up) {
  return a.difference_ns(b,up);
}

inline VertexSet intersection_set(const VertexSet& a, const VertexSet& b) {
  return a & b;
}

inline VertexSet intersection_set(const VertexSet& a, const VertexSet& b, vidType up) {
  return a.intersect(b,up);
}

inline uint64_t intersection_num(const VertexSet& a, const VertexSet& b) {
  return a.get_intersect_num(b);
}

inline uint64_t intersection_num(const VertexSet& a, const VertexSet& b, vidType up) {
  return a.intersect_ns(b, up);
}

inline uint64_t intersection_num_except(const VertexSet& a, const VertexSet& b, vidType ancestor) {
  return a.intersect_except(b, ancestor);
}

inline uint64_t intersection_num_except(const VertexSet& a, const VertexSet& b, vidType ancestorA, vidType ancestorB) {
  return a.intersect_except(b, ancestorA, ancestorB);
}

inline uint64_t intersection_num_bound_except(const VertexSet& a, const VertexSet& b, vidType up, vidType ancestor) {
  return a.intersect_ns_except(b, up, ancestor);
}

inline VertexSet bounded(const VertexSet&a, vidType up) {
  return a.bounded(up);
}

inline vidType set_intersection(const VertexSet &a, const VertexSet &b, VertexSet &c) {
  vidType count = 0;
  vidType idx_l = 0, idx_r = 0;
  while(idx_l < a.size() && idx_r < b.size()) {
    vidType left = a[idx_l];
    vidType right = b[idx_r];
    if(left <= right) idx_l++;
    if(right <= left) idx_r++;
    if(left == right) {
      c.add(left);
      count ++;
    }
  }
  return count;
}

inline vidType set_difference(const VertexSet &a, const VertexSet &b, VertexSet &c) {
  vidType count = 0;
  vidType idx_l = 0, idx_r = 0;
  while(idx_l < a.size() && idx_r < b.size()) {
    vidType left = a[idx_l];
    vidType right = b[idx_r];
    if(left <= right) idx_l++;
    if(right <= left) idx_r++;
    if(left < right && left != b.get_vid()) {
      c.add(left);
      count ++;
    }
  }
  while(idx_l < a.size()) {
    vidType left = a[idx_l];
    idx_l++;
    if(left != b.get_vid()) {
      c.add(left);
      count ++;
    }
  }
  return count;
}

inline vidType intersection_num(VertexSet& vs, VertexList u_begins, VertexList u_ends) {
  int v_size = vs.size();
  int u_size = u_begins.size();
  vidType num = 0;
  int idx_l = 0, idx_r = 0;
  while (idx_l < v_size && idx_r < u_size) {
    auto v = vs[idx_l];
    auto u_begin = u_begins[idx_r];
    if (v < u_begin) {
      idx_l++;
      continue;
    }
    auto u_end = u_ends[idx_r];
    if (v >= u_end) {
      if (v == u_end) idx_l++;
      idx_r++;
      continue;
    }
    if (v >= u_begin && v < u_end) {
      num++;
      idx_l++;
    }
  }
  return num;
}

inline vidType intersection_num(VertexSet& vs, VertexList u_begins, VertexList u_ends, vidType up) {
  int v_size = vs.size();
  int u_size = u_begins.size();
  vidType num = 0;
  int idx_l = 0, idx_r = 0;
  while (idx_l < v_size && idx_r < u_size) {
    auto v = vs[idx_l];
    if (v >= up) break;
    auto u_begin = u_begins[idx_r];
    if (u_begin >= up) break;
    if (v < u_begin) {
      idx_l++;
      continue;
    }
    auto u_end = u_ends[idx_r];
    if (v >= u_end) {
      if (v == u_end) idx_l++;
      idx_r++;
      continue;
    }
    if (v >= u_begin && v < u_end) {
      num++;
      idx_l++;
    }
  }
  return num;
}
/*
inline vidType intersection_num(VertexSet& vs, VertexList u_begins, VertexList u_ends, vidType up) {
  for (auto v : vs) {
    if (v >= up) break;
    for (int i = 0; i < u_size; i++) {
      auto u_begin = u_begins[i];
      if (u_begin >= up) break;
      if (v < u_begin) continue;
      auto u_end = u_ends[i];
      if (v < u_end) {
        num++;
        break;
      }
    }
  }
}
*/
inline vidType intersection_num(VertexList v_begins, VertexList v_ends, VertexList u_begins, VertexList u_ends) {
  vidType num = 0;
  int v_size = v_begins.size();
  int u_size = u_begins.size();

  // compare v_itv and u_itv
  int idx_l = 0, idx_r = 0;
  while (idx_l < v_size && idx_r < u_size) {
    auto v_begin = v_begins[idx_l];
    auto v_end = v_ends[idx_l];
    auto u_begin = u_begins[idx_r];
    auto u_end = u_ends[idx_r];
    assert(v_end > v_begin);
    assert(u_end > u_begin);
    if (v_begin >= u_end) {
      idx_r++;
      continue;
    }
    if (u_begin >= v_end) {
      idx_l++;
      continue;
    }
    if (v_end >= u_end) idx_r++;
    if (v_end <= u_end) idx_l++;
    num += std::min(v_end, u_end) - std::max(v_begin, u_begin);
  }
  return num;
}

inline vidType intersection_num(VertexList v_begins, VertexList v_ends, VertexList u_begins, VertexList u_ends, vidType up) {
  vidType num = 0;
  int v_size = v_begins.size();
  int u_size = u_begins.size();

  // compare v_itv and u_itv
  int idx_l = 0, idx_r = 0;
  while (idx_l < v_size && idx_r < u_size) {
    auto v_begin = v_begins[idx_l];
    auto v_end = v_ends[idx_l];
    auto u_begin = u_begins[idx_r];
    auto u_end = u_ends[idx_r];
    assert(v_end > v_begin);
    assert(u_end > u_begin);
    if (v_begin >= up || u_begin >= up) break;
    if (v_begin >= u_end) {
      idx_r++;
      continue;
    }
    if (u_begin >= v_end) {
      idx_l++;
      continue;
    }
    if (v_end >= u_end) idx_r++;
    if (v_end <= u_end) idx_l++;
    num += std::min(up, std::min(v_end, u_end)) - std::max(v_begin, u_begin);
  }
  return num;
}

