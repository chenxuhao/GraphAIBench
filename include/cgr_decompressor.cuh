#pragma once
#include "common.h"

class CgrReaderGPU {
  public:
    eidType global_offset;
    vidType *graph;
    vidType node;

    __device__ CgrReaderGPU() {}

    __device__ CgrReaderGPU(vidType v, vidType *g, eidType off) :
      node(v), graph(g), global_offset(off) { }

    __device__ void init(vidType node, vidType *graph, eidType global_offset) {
      this->node = node;
      this->graph = graph;
      this->global_offset = global_offset;
    }

    static __device__ vidType decode_first_num(vidType node, vidType x) {
      return (x & 1) ? node - (x >> 1) - 1 : node + (x >> 1);
    }

    __device__ vidType cur() {
      eidType chunk = global_offset / 32;
      //if (threadIdx.x == 0) printf("v %d chunk=%ld, global_offset=%ld\n", node, chunk, global_offset);
      vidType buf_hi = graph[chunk];
      vidType buf_lo = graph[chunk + 1];
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

#if ZETA_K != 1
    __device__ vidType decode_zeta() {
      vidType h = decode_unary();
      global_offset++;
      vidType x = decode_int(h * ZETA_K);
      return x - 1;
    }
#endif

    __device__ vidType decode_residual_code() {
#if ZETA_K == 1
      return decode_gamma();
#else
      return decode_zeta();
#endif
    }

    __device__ vidType decode_segment_cnt() {
      vidType segment_cnt = node == SIZE_NONE ? 0 : decode_gamma() + 1;
      if (segment_cnt == 1 && (cur() & 0x80000000)) {
        global_offset += 1;
        segment_cnt = 0;
      }
      return segment_cnt;
    }
};

struct ResidualSegmentHelperGPU {
  vidType residual_cnt;
  vidType left;
  bool first_res;
  CgrReaderGPU &cgrr;

  __device__ ResidualSegmentHelperGPU(vidType node, CgrReaderGPU &cgrr) :
    cgrr(cgrr), first_res(true), left(0), residual_cnt(0) {
    }

  __device__ void decode_residual_cnt() {
    this->residual_cnt = cgrr.node == SIZE_NONE ? 0 : cgrr.decode_gamma();
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
    return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
  }

};

struct IntervalSegmentHelperGPU {
  vidType interval_cnt;
  vidType left;
  bool first_interval;
  CgrReaderGPU &cgrr;

  __device__ IntervalSegmentHelperGPU(vidType node, CgrReaderGPU &cgrr) :
    cgrr(cgrr), first_interval(true), left(0), interval_cnt(0) {
    }

  __device__ void decode_interval_cnt() {
    interval_cnt = cgrr.node == SIZE_NONE ? 0 : cgrr.decode_gamma();
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
    return (x & 1) ? cgrr.node - (x >> 1) - 1 : cgrr.node + (x >> 1);
  }
};

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

// sequentially decode intervals
__device__ void decode_intervals_naive(CgrReaderGPU &decoder, vidType *adj_out, vidType *num_neighbors) {
  //vidType thread_id = threadIdx.x;
  __shared__ vidType offset;
  auto v = decoder.node;
  auto adj_in = decoder.graph;
  auto segment_cnt = decoder.decode_segment_cnt();
  auto interval_offset = decoder.global_offset + threadIdx.x*INTERVAL_SEGMENT_LEN;
  //if (threadIdx.x == 0 && segment_cnt>1) printf("v %d, interval segment cnt: %d\n", v, segment_cnt);
  auto end = ((segment_cnt+BLOCK_SIZE-1)/BLOCK_SIZE) * BLOCK_SIZE;
  for (vidType i = threadIdx.x; i < end; i+=BLOCK_SIZE) {
    CgrReaderGPU cgrr(v, adj_in, interval_offset);
    if (i < segment_cnt) {
      IntervalSegmentHelperGPU isHelper(v, cgrr);
      isHelper.decode_interval_cnt();
      auto num_intervals = isHelper.interval_cnt;
      //if (threadIdx.x == 0) printf("v %d, interval[%d] size: %d\n", v, i, num_intervals);
      for (vidType j = 0; j < num_intervals; j++) {
        auto left = isHelper.get_interval_left();
        auto len = isHelper.get_interval_len();
        auto index = atomicAdd(num_neighbors, len);
        for (vidType k = 0; k < len; k++)
          adj_out[index++] = left+k;
      }
    }
    if (__syncthreads_or(i >= segment_cnt)) {
      if ((i+1) == segment_cnt) {// last segment
        offset = cgrr.global_offset;
        //printf("v %d, offset after interval: %d\n", v, offset);
      }
      __syncthreads();
      decoder.global_offset = offset;
      break;
    } else {
      interval_offset += INTERVAL_SEGMENT_LEN * BLOCK_SIZE;
    }
  }
  __syncthreads();
}

// sequentially decode intervals
__device__ void decode_intervals_warp_naive(CgrReaderGPU &decoder, vidType *adj_out, vidType *num_neighbors) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ vidType offset[WARPS_PER_BLOCK];
  auto v = decoder.node;
  auto adj_in = decoder.graph;
  auto segment_cnt = decoder.decode_segment_cnt();
  auto interval_offset = decoder.global_offset + thread_lane*INTERVAL_SEGMENT_LEN;
  auto end = ((segment_cnt+WARP_SIZE-1)/WARP_SIZE) * WARP_SIZE;
  for (vidType i = thread_lane; i < end; i+=WARP_SIZE) {
    CgrReaderGPU cgrr(v, adj_in, interval_offset);
    if (i < segment_cnt) {
      IntervalSegmentHelperGPU isHelper(v, cgrr);
      isHelper.decode_interval_cnt();
      auto num_intervals = isHelper.interval_cnt;
      for (vidType j = 0; j < num_intervals; j++) {
        auto left = isHelper.get_interval_left();
        auto len = isHelper.get_interval_len();
        auto index = atomicAdd(num_neighbors, len);
        for (vidType k = 0; k < len; k++)
          adj_out[index++] = left+k;
      }
    }
    int not_done = (i < segment_cnt) ? 1 : 0;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, not_done);
    if (mask != FULL_MASK) {
    //if (__syncthreads_or(i >= segment_cnt)) {
      if ((i+1) == segment_cnt) {// last segment
        offset[warp_lane] = cgrr.global_offset;
      }
      __syncwarp();
      decoder.global_offset = offset[warp_lane];
      break;
    } else {
      interval_offset += INTERVAL_SEGMENT_LEN * WARP_SIZE;
    }
  }
  __syncwarp();
}

// one thread takes one segment
__device__ void decode_residuals_naive(CgrReaderGPU &decoder, vidType *adj_out, vidType *num_neighbors) {
  auto v = decoder.node;
  auto adj_in = decoder.graph;
  auto segment_cnt = decoder.decode_segment_cnt();
  auto residual_offset = decoder.global_offset + threadIdx.x*RESIDUAL_SEGMENT_LEN;
  for (vidType i = threadIdx.x; i < segment_cnt; i+=BLOCK_SIZE) {
    CgrReaderGPU cgrr(v, adj_in, residual_offset);
    ResidualSegmentHelperGPU rsHelper(v, cgrr);
    rsHelper.decode_residual_cnt();
    auto num_res = rsHelper.residual_cnt;
    auto index = atomicAdd(num_neighbors, num_res);
    for (vidType j = 0; j < num_res; j++) {
      auto residual = rsHelper.get_residual();
      adj_out[index++] = residual;
    }
    residual_offset += RESIDUAL_SEGMENT_LEN * BLOCK_SIZE;
  }
  __syncthreads();
}

// one thread takes one segment
__device__ void decode_residuals_warp_naive(CgrReaderGPU &decoder, vidType *adj_out, vidType *num_neighbors) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  auto v = decoder.node;
  auto adj_in = decoder.graph;
  auto segment_cnt = decoder.decode_segment_cnt();
  auto residual_offset = decoder.global_offset + thread_lane*RESIDUAL_SEGMENT_LEN;
  //if (thread_lane == 0 && segment_cnt>1) printf("v %d, redidual segment cnt: %d\n", v, segment_cnt);
  for (vidType i = thread_lane; i < segment_cnt; i+=WARP_SIZE) {
    CgrReaderGPU cgrr(v, adj_in, residual_offset);
    ResidualSegmentHelperGPU rsHelper(v, cgrr);
    rsHelper.decode_residual_cnt();
    auto num_res = rsHelper.residual_cnt;
    auto index = atomicAdd(num_neighbors, num_res);
    //if (thread_lane == 0) printf("v %d, redidual[%d] size: %d\n", v, i, num_res);
    //if (index+num_res > max_degree) printf("v %d, redidual[%d] size: %d, index=%d\n", v, i, num_res, index);
    for (vidType j = 0; j < num_res; j++) {
      auto residual = rsHelper.get_residual();
      //if (thread_lane == 0) printf("v %d, redidual[%d][%d]: %d\n", v, i, j, residual);
      adj_out[index++] = residual;
    }
    residual_offset += RESIDUAL_SEGMENT_LEN * WARP_SIZE;
  }
  __syncwarp();
}


typedef cub::BlockScan<vidType, BLOCK_SIZE> BlockScan;
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

__device__ void handle_one_interval_segment(vidType v, vidType *adj_in, volatile eidType &global_offset, 
                                            SMem *smem, vidType *adj_out, vidType *out_len) {
  vidType thread_id = threadIdx.x;
  CgrReaderGPU cgrr;
  cgrr.init(v, adj_in, global_offset);
  IntervalSegmentHelperGPU sh(v, cgrr);
  sh.decode_interval_cnt();
  vidType thread_data = sh.interval_cnt;
  vidType rsv_rank;
  vidType total;
  __syncthreads();
  BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
  __syncthreads();
  vidType cta_progress = 0;
  while (cta_progress < total) {
    smem->len[thread_id] = 0;
    __syncthreads();
    while ((rsv_rank < cta_progress + BLOCK_SIZE) && (sh.interval_cnt)) {
      smem->left[rsv_rank - cta_progress] = sh.get_interval_left();
      smem->len[rsv_rank - cta_progress] = sh.get_interval_len();
      rsv_rank++;
    }
    __syncthreads();
    auto length = smem->len[thread_id];
    auto index = atomicAdd(out_len, length);
    for (vidType k = 0; k < length; k++) {
      adj_out[index+k] = smem->left[thread_id] + k;
    }
    cta_progress += BLOCK_SIZE;
    __syncthreads();
  }
  global_offset = cgrr.global_offset;
}

__device__ void decode_intervals(CgrReaderGPU &cgrr, SMem *smem, vidType *adj, vidType *out_len) {
  vidType thread_id = threadIdx.x;
  vidType lane_id = thread_id % 32;
  vidType warp_id = thread_id / 32;

  // for retrieve global offset for last segment
  vidType last_segment = SIZE_NONE;
  int segment_cnt = cgrr.decode_segment_cnt();
  // cta gather
  while (__syncthreads_or(segment_cnt >= BLOCK_SIZE)) {
    // vie for control of block
    if (segment_cnt >= BLOCK_SIZE) smem->comm[0][0] = thread_id;
    __syncthreads();
    // winner describes adjlist
    if (smem->comm[0][0] == thread_id) {
      smem->segment_node[0] = cgrr.node;
      smem->segment_offset[0] = cgrr.global_offset;
      segment_cnt -= BLOCK_SIZE;
      cgrr.global_offset += INTERVAL_SEGMENT_LEN * BLOCK_SIZE;
      if (segment_cnt == 0) {
        last_segment = BLOCK_SIZE - 1;
      }
    }
    __syncthreads();
    vidType v = smem->segment_node[0];
    volatile eidType offset = smem->segment_offset[0] + INTERVAL_SEGMENT_LEN * thread_id;
    handle_one_interval_segment(v, cgrr.graph, offset, smem, adj, out_len);
    if (thread_id == BLOCK_SIZE - 1) smem->segment_offset[thread_id] = offset;
    __syncthreads();
    if (last_segment != SIZE_NONE) {
      cgrr.global_offset = smem->segment_offset[last_segment];
      last_segment = SIZE_NONE;
    }
  }

  vidType thread_data = segment_cnt;
  vidType rsv_rank = 0;
  vidType total = 0;
  __syncthreads();
  BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
  __syncthreads();
  vidType cta_progress = 0;
  while (cta_progress < total) {
    smem->segment_node[thread_id] = SIZE_NONE;
    __syncthreads();

    while (__any_sync(FULL_MASK, (rsv_rank + 32 < cta_progress + BLOCK_SIZE) && (segment_cnt >= 32))) {
      // vie for control of warp
      if ((rsv_rank + 32 < cta_progress + BLOCK_SIZE) && (segment_cnt >= 32)) {
        smem->comm[warp_id][0] = lane_id;
      }
      vidType boss_lane = smem->comm[warp_id][0];
      vidType boss_node = __shfl_sync(FULL_MASK, cgrr.node, boss_lane);
      eidType boss_global_offset = __shfl_sync(FULL_MASK, cgrr.global_offset, boss_lane);
      vidType boss_rsv_rank = __shfl_sync(FULL_MASK, rsv_rank, boss_lane);
      smem->segment_node[boss_rsv_rank - cta_progress + lane_id] = boss_node;
      smem->segment_offset[boss_rsv_rank - cta_progress + lane_id] = boss_global_offset + lane_id * INTERVAL_SEGMENT_LEN;
      if (boss_lane == lane_id) {
        rsv_rank += 32;
        segment_cnt -= 32;
        cgrr.global_offset += 32 * INTERVAL_SEGMENT_LEN;

        if (segment_cnt == 0) {
          last_segment = boss_rsv_rank - cta_progress + 31;
        }
      }
    }
    while ((rsv_rank < cta_progress + BLOCK_SIZE) && segment_cnt) {
      smem->segment_offset[rsv_rank - cta_progress] = cgrr.global_offset;
      smem->segment_node[rsv_rank - cta_progress] = cgrr.node;
      segment_cnt--;
      if (0 == segment_cnt) {
        last_segment = rsv_rank - cta_progress;
      }
      rsv_rank++;
      cgrr.global_offset += INTERVAL_SEGMENT_LEN;
    }
    __syncthreads();
    handle_one_interval_segment(smem->segment_node[thread_id], cgrr.graph, smem->segment_offset[thread_id], smem, adj, out_len);
    cta_progress += BLOCK_SIZE;
    __syncthreads();
    if (last_segment != SIZE_NONE) {
      cgrr.global_offset = smem->segment_offset[last_segment];
      last_segment = SIZE_NONE;
    }
  }
}

__device__ void handle_one_residual_segment(vidType v, vidType* adj_in, eidType offset, 
                                            SMem *smem, vidType *adj_out, vidType *out_len) {
  //vidType thread_id = threadIdx.x;
  //vidType lane_id = thread_id % 32;
  //vidType warp_id = thread_id / 32;
  CgrReaderGPU cgrr;
  cgrr.init(v, adj_in, offset);
  ResidualSegmentHelperGPU sh(v, cgrr);
  sh.decode_residual_cnt();
  auto num_res = sh.residual_cnt;
  for (vidType j = 0; j < num_res; j++) {
  //while (sh.residual_cnt) {
  //while (__all_sync(FULL_MASK, sh.residual_cnt)) {
    //auto index = atomicAdd(out_len, 1);
    auto index = (*out_len);
    adj_out[index] = sh.get_residual();
    (*out_len)++;
  }
  /*
    vidType scatter = 0;
    vidType warp_aggregate = 0;
    WarpScan(smem->temp_storage[warp_id]).ExclusiveSum(1, scatter, warp_aggregate);
    if (0 == lane_id) {
      smem->output_warp_offset[warp_id] = atomicAdd(out_len, warp_aggregate);
    }
    adj_out[smem->output_warp_offset[warp_id] + scatter] = sh.get_residual();
  }

  vidType thread_data = sh.residual_cnt;
  vidType rsv_rank = 0;
  vidType total = 0;
  vidType remain = 0;
  WarpScan(smem->temp_storage[warp_id]).ExclusiveSum(thread_data, rsv_rank, total);
  vidType warp_progress = 0;
  while (warp_progress < total) {
    remain = total - warp_progress;
    while ((rsv_rank < warp_progress + 32) && (sh.residual_cnt)) {
      smem->left[warp_id * 32 + rsv_rank - warp_progress] = sh.get_residual();
      rsv_rank++;
    }
    vidType neighbour;
    thread_data = 1;
    if (lane_id < min(remain, 32)) {
      neighbour = smem->left[thread_id];
    }
    vidType scatter;
    vidType warp_aggregate;
    WarpScan(smem->temp_storage[warp_id]).ExclusiveSum(thread_data, scatter, warp_aggregate);
    if (0 == lane_id) {
      smem->output_warp_offset[warp_id] = atomicAdd(out_len, warp_aggregate);
    }
    if (thread_data) {
      adj_out[smem->output_warp_offset[warp_id] + scatter] = neighbour;
    }
    warp_progress += 32;
  }
  */
}

// one warp takes one segment
__device__ void decode_residuals(CgrReaderGPU &cgrr, SMem *smem, vidType *ptr, vidType *out_len) {
  vidType thread_id = threadIdx.x;
  vidType lane_id = thread_id % 32;
  vidType warp_id = thread_id / 32;
  int segment_cnt = cgrr.decode_segment_cnt();
  // cta gather
  while (__syncthreads_or(segment_cnt >= BLOCK_SIZE)) {
    // vie for control of block
    if (segment_cnt >= BLOCK_SIZE) smem->comm[0][0] = thread_id;
    __syncthreads();
    // winner describes adjlist
    if (smem->comm[0][0] == thread_id) {
      smem->segment_node[0] = cgrr.node;
      smem->segment_offset[0] = cgrr.global_offset;
      segment_cnt -= BLOCK_SIZE;
      cgrr.global_offset += RESIDUAL_SEGMENT_LEN * BLOCK_SIZE;
    }
    __syncthreads();
    vidType v = smem->segment_node[0];
    eidType offset = smem->segment_offset[0] + RESIDUAL_SEGMENT_LEN * thread_id;
    handle_one_residual_segment(v, cgrr.graph, offset, smem, ptr, out_len);
  }
  vidType thread_data = segment_cnt;
  vidType rsv_rank = 0;
  vidType total = 0;
  __syncthreads();
  BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
  __syncthreads();
  vidType cta_progress = 0;
  while (cta_progress < total) {
    smem->segment_node[thread_id] = SIZE_NONE;
    __syncthreads();
    while (__any_sync(FULL_MASK, (rsv_rank + 32 < cta_progress + BLOCK_SIZE) && (segment_cnt >= 32))) {
      // vie for control of warp
      if ((rsv_rank + 32 < cta_progress + BLOCK_SIZE) && (segment_cnt >= 32)) {
        smem->comm[warp_id][0] = lane_id;
      }
      vidType boss_lane = smem->comm[warp_id][0];
      vidType boss_node = __shfl_sync(FULL_MASK, cgrr.node, boss_lane);
      eidType boss_global_offset = __shfl_sync(FULL_MASK, cgrr.global_offset, boss_lane);
      vidType boss_rsv_rank = __shfl_sync(FULL_MASK, rsv_rank, boss_lane);
      smem->segment_node[boss_rsv_rank - cta_progress + lane_id] = boss_node;
      smem->segment_offset[boss_rsv_rank - cta_progress + lane_id] = boss_global_offset + lane_id * RESIDUAL_SEGMENT_LEN;
      if (boss_lane == lane_id) {
        rsv_rank += 32;
        segment_cnt -= 32;
        cgrr.global_offset += 32 * RESIDUAL_SEGMENT_LEN;
      }
    }
    while ((rsv_rank < cta_progress + BLOCK_SIZE) && segment_cnt) {
      smem->segment_offset[rsv_rank - cta_progress] = cgrr.global_offset;
      smem->segment_node[rsv_rank - cta_progress] = cgrr.node;
      rsv_rank++;
      segment_cnt--;
      cgrr.global_offset += RESIDUAL_SEGMENT_LEN;
    }
    __syncthreads();
    handle_one_residual_segment(smem->segment_node[thread_id], cgrr.graph, smem->segment_offset[thread_id], smem, ptr, out_len);
    cta_progress += BLOCK_SIZE;
    __syncthreads();
  }
}

