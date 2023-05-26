#pragma once
#include "common.h"
using OFFSET_TYPE = int64_t;

template<typename T = vidType>
class UnaryDecoderGPU {
  protected:
    T *word_array;
    OFFSET_TYPE global_offset;

  public:
    __device__ UnaryDecoderGPU(T *g, OFFSET_TYPE off) : 
      word_array(g), global_offset(off) {
    }
    __device__ OFFSET_TYPE get_offset() { return global_offset; }
    __device__ void set_offset(OFFSET_TYPE off) { global_offset = off; }
    __device__ void inc_offset(OFFSET_TYPE off) { global_offset += off; }
 
    static __device__ vidType decode_first_num(vidType node, vidType x) {
      return (x & 1) ? node - (x >> 1) - 1 : node + (x >> 1);
    }

    __device__ vidType cur() {
      eidType chunk = global_offset / 32;
      //if (threadIdx.x == 0) printf("chunk=%ld, global_offset=%ld\n", chunk, global_offset);
      vidType buf_hi = word_array[chunk];
      vidType buf_lo = word_array[chunk + 1];
      vidType offset = global_offset % 32;
      return __funnelshift_l(buf_lo, buf_hi, offset);
    }

    __device__ vidType decode_unary() {
      vidType tmp = cur();
      vidType x = __clz(tmp);
      global_offset += x;
      return x + 1;
    }

    __device__ vidType decode_int(vidType len) {
      vidType x = cur() >> (32 - len);
      global_offset += len;
      return x;
    }

    __device__ vidType decode_gamma() {
      vidType h = decode_unary();
      return this->decode_int(h) - 1;
    }

    __device__ vidType decode_zeta() {
      vidType h = decode_unary();
      global_offset++;
      vidType x = decode_int(h * ZETA_K);
      return x - 1;
    }

    __device__ vidType decode_residual_code() {
#if ZETA_K == 1
      return decode_gamma();
#else
      return decode_zeta();
#endif
    }
};
