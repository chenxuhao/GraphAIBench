#include "cgr_decoder.cuh"

template <typename T>
__device__ vidType cgr_decoder_gpu<T>::decode() {
  vidType degree = 0;
#if USE_INTERVAL
  degree += decode_intervals_warp(out_ptr);
#endif
  degree += decode_residuals_warp(out_ptr+degree);
  return degree;
}

// sequentially decode intervals
template <typename T>
__device__ void cgr_decoder_gpu<T>::decode_intervals_cta(vidType *adj_out, vidType *num_neighbors) {
  __shared__ vidType offset;
  auto segment_cnt = reader.decode_segment_cnt();
  auto interval_offset = reader.get_offset() + threadIdx.x*INTERVAL_SEGMENT_LEN;
  auto end = ((segment_cnt+BLOCK_SIZE-1)/BLOCK_SIZE) * BLOCK_SIZE;
  for (vidType i = threadIdx.x; i < end; i+=BLOCK_SIZE) {
    CgrReaderGPU<T> cgrr(get_id(), in_ptr, interval_offset);
    if (i < segment_cnt) {
      IntervalSegmentHelperGPU isHelper(cgrr);
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
    if (__syncthreads_or(i >= segment_cnt)) {
      if ((i+1) == segment_cnt) {// last segment
        offset = cgrr.get_offset();
      }
      __syncthreads();
      reader.set_offset(offset);
      break;
    } else {
      interval_offset += INTERVAL_SEGMENT_LEN * BLOCK_SIZE;
    }
  }
  __syncthreads();
}

// sequentially decode intervals and insert vertices into an unordered vertex set; 
// return the total number of vertices
template <typename T>
__device__ vidType cgr_decoder_gpu<T>::decode_intervals_warp(vidType *adj_out) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  //__shared__ typename WarpScan::TempStorage temp_storage[WARPS_PER_BLOCK];
  __shared__ eidType offset[WARPS_PER_BLOCK];
  __shared__ vidType degree[WARPS_PER_BLOCK];
  __shared__ vidType start_idx[WARPS_PER_BLOCK];
  if (thread_lane == 0) offset[warp_lane] = 0;
  if (thread_lane == 0) degree[warp_lane] = 0;
  if (thread_lane == 0) start_idx[warp_lane] = 0;
  __syncwarp();
  auto segment_cnt = reader.decode_segment_cnt();
  auto interval_offset = reader.get_offset() + thread_lane*INTERVAL_SEGMENT_LEN;
  auto end = ((segment_cnt+WARP_SIZE-1)/WARP_SIZE) * WARP_SIZE;
  for (vidType i = thread_lane; i < end; i+=WARP_SIZE) {
    CgrReaderGPU<T> cgrr(get_id(), in_ptr, interval_offset);
    IntervalSegmentHelperGPU isHelper(cgrr);
    vidType num_intervals = 0;
    vidType num_items = 0;
    if (i < segment_cnt) {
      isHelper.decode_interval_cnt();
      num_intervals = isHelper.interval_cnt;
      for (vidType j = 0; j < num_intervals; j++) {
        auto left = isHelper.get_interval_left();
        auto len = isHelper.get_interval_len();
        atomicAdd(&start_idx[warp_lane], len);
        for (vidType k = 0; k < len; k++)
          adj_out[start_idx[warp_lane]+k] = left+k;
        num_items += len;
      }
    }
    atomicAdd(&degree[warp_lane], num_items);
    if ((i+1) == segment_cnt) {// last segment
      offset[warp_lane] = cgrr.get_offset();
    }
    __syncwarp();
    int not_done = (i < segment_cnt-1) ? 1 : 0;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, not_done);
    if (mask != FULL_MASK) {
      reader.set_offset(offset[warp_lane]);
      break;
    } else {
      interval_offset += INTERVAL_SEGMENT_LEN * WARP_SIZE;
    }
  }
  __syncwarp();
  return degree[warp_lane];
}

// sequentially decode intervals
template <typename T>
__device__ vidType cgr_decoder_gpu<T>::decode_intervals_warp(vidType *adj_out, vidType &total_num_itvs) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ typename WarpScan::TempStorage temp_storage[WARPS_PER_BLOCK];
  __shared__ eidType offset[WARPS_PER_BLOCK];
  __shared__ vidType degree[WARPS_PER_BLOCK];
  __shared__ vidType start_idx[WARPS_PER_BLOCK];
  if (thread_lane == 0) offset[warp_lane] = 0;
  if (thread_lane == 0) degree[warp_lane] = 0;
  if (thread_lane == 0) start_idx[warp_lane] = 0;
  __syncwarp();
  auto v = get_id();
  auto segment_cnt = reader.decode_segment_cnt();
  //if (segment_cnt > 0 && thread_lane == 0) printf("vertex %u has %u interval segments: \n", v, segment_cnt);
  auto interval_offset = reader.get_offset() + thread_lane*INTERVAL_SEGMENT_LEN;
  auto end = ((segment_cnt+WARP_SIZE-1)/WARP_SIZE) * WARP_SIZE;
  for (vidType i = thread_lane; i < end; i+=WARP_SIZE) {
    CgrReaderGPU<T> cgrr(v, in_ptr, interval_offset);
    IntervalSegmentHelperGPU isHelper(cgrr);
    vidType num_intervals = 0;
    vidType num_items = 0;
    if (i < segment_cnt) {
      isHelper.decode_interval_cnt();
      num_intervals = isHelper.interval_cnt;
      //printf("\t vertex %u segment %u has %u intervals: \n", v, i, num_intervals);
    }
    vidType index = 0, total = 0;
    WarpScan(temp_storage[warp_lane]).ExclusiveSum(2*num_intervals, index, total);
    if (i < segment_cnt) {
      for (vidType j = 0; j < num_intervals; j++) {
        auto left = isHelper.get_interval_left();
        auto len = isHelper.get_interval_len();
        adj_out[start_idx[warp_lane]+index++] = left;
        adj_out[start_idx[warp_lane]+index++] = len;
        num_items += len;
        //printf("\t\t vertex %u interval[%u]: left=%u, len=%u\n", v, j, lefts[threadIdx.x][j], lens[threadIdx.x][j]);
      }
    }
    if (thread_lane == 0) start_idx[warp_lane] += total;
    atomicAdd(&degree[warp_lane], num_items);
    if ((i+1) == segment_cnt) {// last segment
      offset[warp_lane] = cgrr.get_offset();
    }
    __syncwarp();
    int not_done = (i < segment_cnt-1) ? 1 : 0;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, not_done);
    if (mask != FULL_MASK) {
      reader.set_offset(offset[warp_lane]);
      break;
    } else {
      interval_offset += INTERVAL_SEGMENT_LEN * WARP_SIZE;
    }
  }
  __syncwarp();
  total_num_itvs = start_idx[warp_lane] / 2;
  return degree[warp_lane];
}

// sequentially decode intervals
template <typename T>
__device__ vidType cgr_decoder_gpu<T>::decode_intervals_warp(vidType *begins, vidType *ends) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ eidType offset[WARPS_PER_BLOCK];
  __shared__ vidType start_idx[WARPS_PER_BLOCK];
  if (thread_lane == 0) offset[warp_lane] = 0;
  if (thread_lane == 0) start_idx[warp_lane] = 0;
  __syncwarp();
  __shared__ typename WarpScan::TempStorage temp_storage[WARPS_PER_BLOCK];

  auto segment_cnt = reader.decode_segment_cnt();
  auto interval_offset = reader.get_offset() + thread_lane*INTERVAL_SEGMENT_LEN;
  auto end = ((segment_cnt+WARP_SIZE-1)/WARP_SIZE) * WARP_SIZE;
  for (vidType i = thread_lane; i < end; i+=WARP_SIZE) {
    CgrReaderGPU<T> cgrr(get_id(), in_ptr, interval_offset);
    IntervalSegmentHelperGPU isHelper(cgrr);
    vidType num_intervals = 0;
    if (i < segment_cnt) {
      isHelper.decode_interval_cnt();
      num_intervals = isHelper.interval_cnt;
    }
    vidType index = 0, total = 0;
    WarpScan(temp_storage[warp_lane]).ExclusiveSum(num_intervals, index, total);
    if (i < segment_cnt) {
      for (vidType j = 0; j < num_intervals; j++) {
        auto left = isHelper.get_interval_left();
        auto len = isHelper.get_interval_len();
        begins[start_idx[warp_lane]+index+j] = left;
        ends[start_idx[warp_lane]+index+j] = left+len;
      }
    }
    if (thread_lane == 0) start_idx[warp_lane] += total;
    if ((i+1) == segment_cnt) // last segment
      offset[warp_lane] = cgrr.get_offset();
    __syncwarp();
    int not_done = (i < segment_cnt-1) ? 1 : 0;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, not_done);
    if (mask != FULL_MASK) {
      reader.set_offset(offset[warp_lane]);
      break;
    } else {
      interval_offset += INTERVAL_SEGMENT_LEN * WARP_SIZE;
    }
  }
  __syncwarp();
  return start_idx[warp_lane];
}

// one thread takes one segment
template <typename T>
__device__ void cgr_decoder_gpu<T>::decode_residuals_cta(vidType *adj_out, vidType *num_neighbors) {
  auto segment_cnt = reader.decode_segment_cnt();
  auto residual_offset = reader.get_offset() + threadIdx.x*RESIDUAL_SEGMENT_LEN;
  for (vidType i = threadIdx.x; i < segment_cnt; i+=BLOCK_SIZE) {
    CgrReaderGPU<T> cgrr(get_id(), in_ptr, residual_offset);
    ResidualSegmentHelperGPU rsHelper(cgrr);
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
template <typename T>
__device__ vidType cgr_decoder_gpu<T>::decode_residuals_warp(vidType *adj_out) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ typename WarpScan::TempStorage temp_storage[WARPS_PER_BLOCK];
  __shared__ vidType degree[WARPS_PER_BLOCK];
  if (thread_lane == 0) degree[warp_lane] = 0;
  vidType start_idx = 0;
 
  auto segment_cnt = reader.decode_segment_cnt();
  auto residual_offset = reader.get_offset() + thread_lane*RESIDUAL_SEGMENT_LEN;
  auto end = ((segment_cnt+WARP_SIZE-1)/WARP_SIZE) * WARP_SIZE;
  for (vidType i = thread_lane; i < end; i+=WARP_SIZE) {
    CgrReaderGPU<T> cgrr(get_id(), in_ptr, residual_offset);
    ResidualSegmentHelperGPU rsHelper(cgrr);
    vidType num_res = 0, index = 0, sum = 0;
    if (i < segment_cnt) {
      rsHelper.decode_residual_cnt();
      num_res = rsHelper.residual_cnt;
    }
    WarpScan(temp_storage[warp_lane]).ExclusiveSum(num_res, index, sum);
    for (vidType j = 0; j < num_res; j++) {
      auto residual = rsHelper.get_residual();
      adj_out[start_idx+index+j] = residual;
    }
    if (thread_lane == 0) degree[warp_lane] += sum;
    __syncwarp();
    start_idx += sum;
    __syncwarp();
    residual_offset += RESIDUAL_SEGMENT_LEN * WARP_SIZE;
  }
  return degree[warp_lane];
}

template <typename T>
__device__ void cgr_decoder_gpu<T>::handle_one_interval_segment(vidType v,
                                                                vidType *adj_in,
                                                                volatile eidType &global_offset, 
                                                                SMem *smem,
                                                                vidType *adj_out,
                                                                vidType *out_len) {
  vidType thread_id = threadIdx.x;
  CgrReaderGPU<T> cgrr;
  cgrr.init(v, adj_in, global_offset);
  IntervalSegmentHelperGPU sh(cgrr);
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
  global_offset = cgrr.get_offset();
}

template <typename T>
__device__ void cgr_decoder_gpu<T>::decode_intervals(SMem *smem, vidType *adj_out, vidType *out_len) {
  vidType thread_id = threadIdx.x;
  vidType lane_id = thread_id % 32;
  vidType warp_id = thread_id / 32;

  // for retrieve global offset for last segment
  vidType last_segment = SIZE_NONE;
  int segment_cnt = reader.decode_segment_cnt();
  // cta gather
  while (__syncthreads_or(segment_cnt >= BLOCK_SIZE)) {
    // vie for control of block
    if (segment_cnt >= BLOCK_SIZE) smem->comm[0][0] = thread_id;
    __syncthreads();
    // winner describes adjlist
    if (smem->comm[0][0] == thread_id) {
      smem->segment_node[0] = get_id();
      smem->segment_offset[0] = reader.get_offset();
      segment_cnt -= BLOCK_SIZE;
      reader.inc_offset(INTERVAL_SEGMENT_LEN * BLOCK_SIZE);
      if (segment_cnt == 0) {
        last_segment = BLOCK_SIZE - 1;
      }
    }
    __syncthreads();
    vidType v = smem->segment_node[0];
    volatile eidType offset = smem->segment_offset[0] + INTERVAL_SEGMENT_LEN * thread_id;
    handle_one_interval_segment(v, in_ptr, offset, smem, adj_out, out_len);
    if (thread_id == BLOCK_SIZE - 1) smem->segment_offset[thread_id] = offset;
    __syncthreads();
    if (last_segment != SIZE_NONE) {
      reader.set_offset(smem->segment_offset[last_segment]);
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
      vidType boss_node = __shfl_sync(FULL_MASK, get_id(), boss_lane);
      eidType boss_global_offset = __shfl_sync(FULL_MASK, reader.get_offset(), boss_lane);
      vidType boss_rsv_rank = __shfl_sync(FULL_MASK, rsv_rank, boss_lane);
      smem->segment_node[boss_rsv_rank - cta_progress + lane_id] = boss_node;
      smem->segment_offset[boss_rsv_rank - cta_progress + lane_id] = boss_global_offset + lane_id * INTERVAL_SEGMENT_LEN;
      if (boss_lane == lane_id) {
        rsv_rank += 32;
        segment_cnt -= 32;
        reader.inc_offset(32 * INTERVAL_SEGMENT_LEN);

        if (segment_cnt == 0) {
          last_segment = boss_rsv_rank - cta_progress + 31;
        }
      }
    }
    while ((rsv_rank < cta_progress + BLOCK_SIZE) && segment_cnt) {
      smem->segment_offset[rsv_rank - cta_progress] = reader.get_offset();
      smem->segment_node[rsv_rank - cta_progress] = get_id();
      segment_cnt--;
      if (0 == segment_cnt) {
        last_segment = rsv_rank - cta_progress;
      }
      rsv_rank++;
      reader.inc_offset(INTERVAL_SEGMENT_LEN);
    }
    __syncthreads();
    handle_one_interval_segment(smem->segment_node[thread_id], in_ptr, smem->segment_offset[thread_id], smem, adj_out, out_len);
    cta_progress += BLOCK_SIZE;
    __syncthreads();
    if (last_segment != SIZE_NONE) {
      reader.set_offset(smem->segment_offset[last_segment]);
      last_segment = SIZE_NONE;
    }
  }
}

template <typename T>
__device__ void cgr_decoder_gpu<T>::handle_one_residual_segment(vidType v, vidType* adj_in, eidType offset, 
                                                             SMem *smem, vidType *adj_out, vidType *out_len) {
  //vidType thread_id = threadIdx.x;
  //vidType lane_id = thread_id % 32;
  //vidType warp_id = thread_id / 32;
  CgrReaderGPU<T> cgrr;
  cgrr.init(v, adj_in, offset);
  ResidualSegmentHelperGPU sh(cgrr);
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
template <typename T>
__device__ void cgr_decoder_gpu<T>::decode_residuals(SMem *smem, vidType *ptr, vidType *out_len) {
  vidType thread_id = threadIdx.x;
  vidType lane_id = thread_id % 32;
  vidType warp_id = thread_id / 32;
  int segment_cnt = reader.decode_segment_cnt();
  // cta gather
  while (__syncthreads_or(segment_cnt >= BLOCK_SIZE)) {
    // vie for control of block
    if (segment_cnt >= BLOCK_SIZE) smem->comm[0][0] = thread_id;
    __syncthreads();
    // winner describes adjlist
    if (smem->comm[0][0] == thread_id) {
      smem->segment_node[0] = get_id();
      smem->segment_offset[0] = reader.get_offset();
      segment_cnt -= BLOCK_SIZE;
      reader.inc_offset(RESIDUAL_SEGMENT_LEN * BLOCK_SIZE);
    }
    __syncthreads();
    vidType v = smem->segment_node[0];
    eidType offset = smem->segment_offset[0] + RESIDUAL_SEGMENT_LEN * thread_id;
    handle_one_residual_segment(v, in_ptr, offset, smem, ptr, out_len);
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
      vidType boss_node = __shfl_sync(FULL_MASK, get_id(), boss_lane);
      eidType boss_global_offset = __shfl_sync(FULL_MASK, reader.get_offset(), boss_lane);
      vidType boss_rsv_rank = __shfl_sync(FULL_MASK, rsv_rank, boss_lane);
      smem->segment_node[boss_rsv_rank - cta_progress + lane_id] = boss_node;
      smem->segment_offset[boss_rsv_rank - cta_progress + lane_id] = boss_global_offset + lane_id * RESIDUAL_SEGMENT_LEN;
      if (boss_lane == lane_id) {
        rsv_rank += 32;
        segment_cnt -= 32;
        reader.inc_offset(32 * RESIDUAL_SEGMENT_LEN);
      }
    }
    while ((rsv_rank < cta_progress + BLOCK_SIZE) && segment_cnt) {
      smem->segment_offset[rsv_rank - cta_progress] = reader.get_offset();
      smem->segment_node[rsv_rank - cta_progress] = get_id();
      rsv_rank++;
      segment_cnt--;
      reader.inc_offset(RESIDUAL_SEGMENT_LEN);
    }
    __syncthreads();
    handle_one_residual_segment(smem->segment_node[thread_id], in_ptr, smem->segment_offset[thread_id], smem, ptr, out_len);
    cta_progress += BLOCK_SIZE;
    __syncthreads();
  }
}

template class cgr_decoder_gpu<vidType>;
