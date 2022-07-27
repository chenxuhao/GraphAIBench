#pragma once

#include "common.h"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/util_allocator.cuh>
#define ZETA_K 3
#define MIN_ITV_LEN 4
//#define RES_SEG_LEN 256
#define RESIDUAL_SEGMENT_LEN 256
#define CTA_SIZE 256

using OFFSET_TYPE = uint64_t;
using SIZE_TYPE = uint32_t;
static const SIZE_TYPE SIZE_NONE = 0xffffffff;

using MASK_TYPE = uint8_t;
const int MASK_LEN = 8;

using GRAPH_TYPE = uint32_t;
const int GRAPH_BYTE = 4;
const int GRAPH_LEN = 32;

const SIZE_TYPE THREADS_NUM = CTA_SIZE;

const SIZE_TYPE INTERVAL_SEGMENT_LEN = RESIDUAL_SEGMENT_LEN ? (8 * 32) : 0;

using hCG = thrust::host_vector<GRAPH_TYPE>;
using hOS = thrust::host_vector<OFFSET_TYPE>;

using dCG = thrust::device_vector<GRAPH_TYPE>;
using dOS = thrust::device_vector<OFFSET_TYPE>;

#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

#define __dsync__ CubDebugExit(cudaDeviceSynchronize())

cub::CachingDeviceAllocator g_allocator(true);

int load_compressed_graph(std::string file_path, hCG &hcg, hOS &hos) {
  // load graph
  std::ifstream ifs;
  ifs.open(file_path + ".graph", std::ios::in | std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    std::cout << "open graph file failed!" << std::endl;
    return -1;
  }
  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  ifs.read((char*) buffer.data(), size);
  hcg.clear();
  GRAPH_TYPE tmp = 0;
  for (size_t i = 0; i < buffer.size(); i++) {
    tmp <<= 8;
    tmp += buffer[i];
    if ((i + 1) % GRAPH_BYTE == 0) {
      hcg.push_back(tmp);
    }
  }
  if (size % GRAPH_BYTE) {
    int rem = size % GRAPH_BYTE;
    while (rem % GRAPH_BYTE)
      tmp <<= 8, rem++;
    hcg.push_back(tmp);
  }

  ifs.close();

  // load offset
  SIZE_TYPE num_node;
  hos.clear();
  hos.push_back(0);
  std::ifstream ifs_offset;
  ifs_offset.open(file_path + ".offset", std::ios::in);
  ifs_offset >> num_node;
  OFFSET_TYPE cur;
  for (auto i = 0; i < num_node; i++) {
    ifs_offset >> cur;
    hos.push_back(cur);
  }
  ifs_offset.close();

  return num_node;
}

class CgrReader {
  public:
    OFFSET_TYPE global_offset;
    GRAPH_TYPE *graph;
    SIZE_TYPE node;

    __device__
      void init(SIZE_TYPE node, GRAPH_TYPE *graph, OFFSET_TYPE global_offset) {
        this->node = node;
        this->graph = graph;
        this->global_offset = global_offset;
      }

    static __device__
      SIZE_TYPE decode_first_num(SIZE_TYPE node, SIZE_TYPE x) {
        return (x & 1) ? node - (x >> 1) - 1 : node + (x >> 1);
      }

    __device__
      GRAPH_TYPE cur() {
        OFFSET_TYPE chunk = global_offset / GRAPH_LEN;
        SIZE_TYPE buf_hi = graph[chunk];
        SIZE_TYPE buf_lo = graph[chunk + 1];
        SIZE_TYPE offset = global_offset % GRAPH_LEN;
        return __funnelshift_l(buf_lo, buf_hi, offset);
      }

    __device__
      SIZE_TYPE decode_unary() {
        SIZE_TYPE tmp = cur();
        SIZE_TYPE x = __clz(tmp);
        global_offset += x;
        return x + 1;
      }

    __device__
      SIZE_TYPE decode_int(SIZE_TYPE len) {
        SIZE_TYPE x = cur() >> (32 - len);
        global_offset += len;
        return x;
      }

    __device__
      SIZE_TYPE decode_gamma() {
        SIZE_TYPE h = decode_unary();
        return this->decode_int(h) - 1;
      }

#if ZETA_K != 1
    __device__
      SIZE_TYPE decode_zeta() {
        SIZE_TYPE h = decode_unary();
        global_offset++;
        SIZE_TYPE x = decode_int(h * ZETA_K);
        return x - 1;
      }
#endif

    __device__
      SIZE_TYPE decode_residual_code() {
#if ZETA_K == 1
        return decode_gamma();
#else
        return decode_zeta();
#endif
      }

    __device__
      SIZE_TYPE decode_segment_cnt() {
        SIZE_TYPE segment_cnt = node == SIZE_NONE ? 0 : decode_gamma() + 1;

        if (segment_cnt == 1 && (cur() & 0x80000000)) {
          global_offset += 1;
          segment_cnt = 0;
        }

        return segment_cnt;
      }
};

struct ResidualSegmentHelper{
  SIZE_TYPE residual_cnt;

  SIZE_TYPE left;
  bool first_res;

  CgrReader &cgrr;

  __device__
    ResidualSegmentHelper(SIZE_TYPE node, CgrReader &cgrr) :
      cgrr(cgrr), first_res(true), left(0), residual_cnt(0) {
      }

  __device__
    void decode_residual_cnt() {
      this->residual_cnt = cgrr.node == SIZE_NONE ? 0 : cgrr.decode_gamma();
    }

  __device__
    SIZE_TYPE get_residual() {
      if (first_res) {
        left = decode_first_num();
        first_res = false;
      } else {
        left += cgrr.decode_residual_code() + 1;
      }
      residual_cnt--;
      return left;
    }

  __device__
    SIZE_TYPE decode_first_num() {
      SIZE_TYPE x = cgrr.decode_residual_code();
      return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
    }

};

struct IntervalSegmentHelper {
  SIZE_TYPE interval_cnt;

  SIZE_TYPE left;
  bool first_interval;

  CgrReader &cgrr;

  __device__
    IntervalSegmentHelper(SIZE_TYPE node, CgrReader &cgrr) :
      cgrr(cgrr), first_interval(true), left(0), interval_cnt(0) {
      }

  __device__
    void decode_interval_cnt() {
      interval_cnt = cgrr.node == SIZE_NONE ? 0 : cgrr.decode_gamma();
    }

  __device__
    SIZE_TYPE get_interval_left() {
      if (first_interval) {
        left = decode_first_num();
        first_interval = false;
      } else {
        left += cgrr.decode_gamma() + 1;
      }
      return left;
    }

  __device__
    SIZE_TYPE get_interval_len() {
      SIZE_TYPE len = cgrr.decode_gamma() + MIN_ITV_LEN;
      left += len;
      interval_cnt--;
      return len;
    }

  __device__
    SIZE_TYPE decode_first_num() {
      SIZE_TYPE x = cgrr.decode_gamma();
      return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
    }
};

struct SeriesHelper {
  SIZE_TYPE interval_num;

  SIZE_TYPE node;
  SIZE_TYPE dout;

  SIZE_TYPE left;
  bool first_res;
  bool first_interval;

  CgrReader &curp;

  __device__ SeriesHelper(SIZE_TYPE node, CgrReader &curp, SIZE_TYPE dout) :
    node(node), curp(curp), dout(dout), first_res(true), first_interval(true) {

      interval_num = dout ? curp.decode_gamma() : 0;
    }

  __device__
    SIZE_TYPE get_interval_left() {
      if (first_interval) {
        left = curp.decode_gamma();
        left = curp.decode_first_num(node, left);
        first_interval = false;
      } else {
        left += curp.decode_gamma() + 1;
      }
      return left;
    }

  __device__
    SIZE_TYPE get_interval_len() {
      SIZE_TYPE len = curp.decode_gamma() + MIN_ITV_LEN;
      dout -= len;
      left += len;
      return len;
    }

  __device__
    SIZE_TYPE get_residual() {
      if (first_res) {
        left = curp.decode_residual_code();
        left = curp.decode_first_num(node, left);
        first_res = false;
      } else {
        left += curp.decode_residual_code() + 1;
      }
      dout--;
      return left;
    }

  __device__
    SIZE_TYPE calc_residual(SIZE_TYPE x) {
      if (first_res) {
        left = x;
        left = curp.decode_first_num(node, left);
        first_res = false;
      } else {
        left += x + 1;
      }
      dout--;
      return left;
    }
};

struct BaseHelper {
  SIZE_TYPE interval_idx;
  SIZE_TYPE interval_num;

  SIZE_TYPE node;
  SIZE_TYPE dout;

  SIZE_TYPE left;
  SIZE_TYPE len ;

  bool first_res;

  CgrReader &curp;

  __device__
    BaseHelper (SIZE_TYPE node, CgrReader &curp, SIZE_TYPE dout) : node(node), curp(curp), dout(dout) {
      if (dout) {
        interval_num = curp.decode_gamma();
        interval_idx = 0;
        len = 0;
        refresh_interval();
        first_res = true;
      }
    }

  __device__
    void refresh_interval() {

      if (interval_idx >= interval_num) return;
      if (len) return;

      if (interval_idx == 0) {
        left = curp.decode_gamma();
        left = curp.decode_first_num(node, left);
      } else {
        left += curp.decode_gamma() + 1;
      }

      len = curp.decode_gamma() + MIN_ITV_LEN;

      interval_idx++;
    }

  __device__
    SIZE_TYPE fetch_next() {
      dout--;
      if (len) {
        // interval
        SIZE_TYPE cur = left;
        left++;
        len--;
        refresh_interval();
        return cur;
      } else {
        // residual
        if (first_res) {
          left = curp.decode_residual_code();
          left = curp.decode_first_num(node, left);
          first_res = false;
          return left;
        } else {
          left += curp.decode_residual_code() + 1;
          return left;
        }
      }
    }
};
