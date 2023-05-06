#pragma once
#include <cub/cub.cuh>
#include "unary_decoder.cuh"

template<typename T = vidType>
class CgrReaderGPU : public UnaryDecoderGPU<T> {
    T id_;
  public:
    __device__ CgrReaderGPU() : UnaryDecoderGPU<T>(NULL, 0) {}
    __device__ CgrReaderGPU(T id, T *data, eidType off) :
      UnaryDecoderGPU<T>(data, off), id_(id) { }
    inline __device__ T get_id() { return id_; }
    __device__ void init(T id, T* data, OFFSET_TYPE off) {
      this->id_ = id;
      this->word_array = data;
      this->global_offset = off;
    }
    inline __device__ T decode_segment_cnt() {
      T segment_cnt = id_ == SIZE_NONE ? 0 : this->decode_gamma() + 1;
      if (segment_cnt == 1 && (this->cur() & 0x80000000)) {
        this->global_offset += 1;
        segment_cnt = 0;
      }
      return segment_cnt;
    }
};

typedef cub::BlockScan<vidType, BLOCK_SIZE> BlockScan;
typedef cub::BlockScan<int, BLOCK_SIZE> BlockScanInt;
typedef cub::WarpScan<vidType> WarpScan;
struct SMem {
  typename BlockScan::TempStorage block_temp_storage;
  typename WarpScan::TempStorage temp_storage[BLOCK_SIZE / 32];
  volatile vidType segment_node[BLOCK_SIZE];
  volatile eidType segment_offset[BLOCK_SIZE];
  volatile vidType left[BLOCK_SIZE];
  volatile vidType len[BLOCK_SIZE];
  volatile vidType comm[BLOCK_SIZE / 32][32];
  volatile vidType output_cta_offset;
  volatile vidType output_warp_offset[BLOCK_SIZE / 32];
};

template <typename T>
class cgr_decoder_gpu {
  private:
    CgrReaderGPU<T> reader;
    T *in_ptr;
    T *out_ptr;
  public:
    __device__ cgr_decoder_gpu(T id, T *in, OFFSET_TYPE off, T* out = NULL) {
      in_ptr = in;
      out_ptr = out;
      #ifdef WORD_ALIGHED 
        reader.init(id, in, off*32); // transform word offset to bit offset
      #elif BYTE_ALIGHED 
        reader.init(id, in, off*8); // transform byte offset to bit offset
      #else
        reader.init(id, in, off);
      #endif
    }
    __device__ vidType decode();
    __device__ T get_id() { return reader.get_id(); }
    __device__ OFFSET_TYPE get_offset() { return reader.get_offset(); }
    __device__ void decode_intervals_cta(vidType *adj_out, vidType *num_neighbors);
    __device__ vidType decode_intervals_warp(vidType *adj_out);
    __device__ vidType decode_intervals_warp(vidType *adj_out, vidType &total_num_itvs);
    __device__ vidType decode_intervals_warp(vidType *begins, vidType *ends);
    __device__ void decode_residuals_cta(vidType *adj_out, vidType *num_neighbors);
    __device__ vidType decode_residuals_warp(vidType *adj_out);
    __device__ void decode_intervals(SMem *smem, vidType *adj_out, vidType *out_len);
    __device__ void decode_residuals(SMem *smem, vidType *ptr, vidType *out_len);

  protected:
    __device__ void handle_one_interval_segment(vidType v, vidType *adj_in, volatile eidType &global_offset, 
                                                SMem *smem, vidType *adj_out, vidType *out_len);
    __device__ void handle_one_residual_segment(vidType v, vidType* adj_in, eidType offset, 
                                                SMem *smem, vidType *adj_out, vidType *out_len);
};

template<typename T = vidType>
struct ResidualSegmentHelperGPU {
  vidType residual_cnt;
  vidType left;
  bool first_res;
  CgrReaderGPU<T> &cgrr;

  __device__ ResidualSegmentHelperGPU(CgrReaderGPU<T> &cgrr) :
    cgrr(cgrr), first_res(true), left(0), residual_cnt(0) { }
  __device__ void decode_residual_cnt() {
    this->residual_cnt = cgrr.get_id() == SIZE_NONE ? 0 : cgrr.decode_gamma();
  }
  __device__ vidType get_residual() {
    if (first_res) {
      left = decode_first_num();
      first_res = false;
    } else {
      left += cgrr.decode_residual_code() + 1;
    }
    residual_cnt--;
    return left;
  }
  __device__ vidType decode_first_num() {
    vidType x = cgrr.decode_residual_code();
    return (x & 1) ? cgrr.get_id() - (x >> 1) - 1 : cgrr.get_id() + (x >> 1);
  }
};

template<typename T = vidType>
struct IntervalSegmentHelperGPU {
  vidType interval_cnt;
  vidType left;
  bool first_interval;
  CgrReaderGPU<T> &cgrr;

  __device__ IntervalSegmentHelperGPU(CgrReaderGPU<T> &cgrr) :
    cgrr(cgrr), first_interval(true), left(0), interval_cnt(0) {
    }

  __device__ void decode_interval_cnt() {
    interval_cnt = cgrr.get_id() == SIZE_NONE ? 0 : cgrr.decode_gamma();
  }

  __device__ vidType get_interval_left() {
    if (first_interval) {
      left = decode_first_num();
      first_interval = false;
    } else {
      left += cgrr.decode_gamma() + 1;
    }
    return left;
  }

  __device__ vidType get_interval_len() {
    vidType len = cgrr.decode_gamma() + MIN_ITV_LEN;
    left += len;
    interval_cnt--;
    return len;
  }

  __device__ vidType decode_first_num() {
    vidType x = cgrr.decode_gamma();
    return (x & 1) ? cgrr.get_id() - (x >> 1) - 1 : cgrr.get_id() + (x >> 1);
  }
};

/*
struct SeriesHelperGPU {
  vidType interval_num;
  vidType node;
  vidType dout;
  vidType left;
  bool first_res;
  bool first_interval;
  CgrReaderGPU &curp;

  __device__ SeriesHelperGPU(vidType node, CgrReaderGPU &curp, vidType dout) :
    node(node), curp(curp), dout(dout), first_res(true), first_interval(true) {
      interval_num = dout ? curp.decode_gamma() : 0;
    }

  __device__ vidType get_interval_left() {
    if (first_interval) {
      left = curp.decode_gamma();
      left = curp.decode_first_num(node, left);
      first_interval = false;
    } else {
      left += curp.decode_gamma() + 1;
    }
    return left;
  }

  __device__ vidType get_interval_len() {
    vidType len = curp.decode_gamma() + MIN_ITV_LEN;
    dout -= len;
    left += len;
    return len;
  }

  __device__ vidType get_residual() {
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

  __device__ vidType calc_residual(vidType x) {
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

struct BaseHelperGPU {
  vidType interval_idx;
  vidType interval_num;
  vidType node;
  vidType dout;
  vidType left;
  vidType len ;
  bool first_res;
  CgrReaderGPU &curp;

  __device__ BaseHelperGPU (vidType node, CgrReaderGPU &curp, vidType dout) :
    node(node), curp(curp), dout(dout) {
      if (dout) {
        interval_num = curp.decode_gamma();
        interval_idx = 0;
        len = 0;
        refresh_interval();
        first_res = true;
      }
    }

  __device__ void refresh_interval() {
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

  __device__ vidType fetch_next() {
    dout--;
    if (len) {
      // interval
      vidType cur = left;
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
*/

