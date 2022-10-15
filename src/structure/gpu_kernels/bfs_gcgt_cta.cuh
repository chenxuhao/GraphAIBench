#pragma once
#include "common.h"
//#define RESIDUAL_SEGMENT_LEN 256
//const vidType INTERVAL_SEGMENT_LEN = RESIDUAL_SEGMENT_LEN ? (8 * 32) : 0;
const int MASK_LEN = 8;
const vidType THREADS_NUM = 256;

template<int THREADS_NUM>
struct BfsGcgtCta {
    typedef cub::BlockScan<vidType, THREADS_NUM> BlockScan;
    typedef cub::WarpScan<vidType> WarpScan;

    struct SMem {
        typename BlockScan::TempStorage block_temp_storage;
        typename WarpScan::TempStorage temp_storage[THREADS_NUM / 32];

        volatile vidType segment_node[THREADS_NUM];
        volatile eidType segment_offset[THREADS_NUM];

        volatile vidType left[THREADS_NUM];
        volatile vidType len[THREADS_NUM];

        volatile vidType comm[THREADS_NUM / 32][32];

        volatile vidType output_cta_offset;
        volatile vidType output_warp_offset[THREADS_NUM / 32];
    };

    vidType iteration;
    vidType src;
    vidType *d_in_len;
    vidType *d_in;
    vidType *d_out_len;
    vidType *d_out;
    eidType *offsets;
    vidType *graph;
    mask_t *visited_mask;
    vidType *labels;
    SMem *smem;

    vidType thread_id;
    vidType lane_id;
    vidType warp_id;

    __device__ BfsGcgtCta(SMem *smem, vidType iteration, vidType src, vidType *d_in_len, vidType *d_in,
            vidType *d_out_len, vidType *d_out, eidType *offsets, vidType *graph, mask_t *visited_mask,
            vidType *labels) :
            smem(smem), iteration(iteration), src(src), d_in_len(d_in_len), d_in(d_in), d_out_len(d_out_len), d_out(
                    d_out), offsets(offsets), graph(graph), visited_mask(visited_mask), labels(labels) {
        thread_id = threadIdx.x;
        lane_id = thread_id % 32;
        warp_id = thread_id / 32;
    }

    __device__
    void interval_cta_gather(vidType &left, vidType &len) {

        while (__syncthreads_or(len >= THREADS_NUM)) {
            // vie for control of block
            if (len >= THREADS_NUM)
                smem->comm[0][0] = thread_id;
            __syncthreads();

            // winner describes adjlist
            if (smem->comm[0][0] == thread_id) {
                smem->comm[0][1] = left;
                smem->comm[0][2] = left + len / THREADS_NUM * THREADS_NUM;
                left += len - len % THREADS_NUM;
                len %= THREADS_NUM;
            }
            __syncthreads();

            vidType gather = smem->comm[0][1] + thread_id;
            vidType gather_end = smem->comm[0][2];
            vidType neighbour;
            vidType thread_data_in;
            vidType thread_data_out;
            vidType block_aggregate;
            
            while (__syncthreads_or(gather < gather_end)) {
                neighbour = gather;
                if (gather < gather_end) {
                    thread_data_in = unvisited(neighbour) ? 1 : 0;
                } else
                    thread_data_in = 0;

                __syncthreads();
                BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data_in, thread_data_out, block_aggregate);
                __syncthreads();
                if (0 == thread_id) {
                    smem->output_cta_offset = atomicAdd(d_out_len, block_aggregate);
                }
                __syncthreads();
                if (thread_data_in) {
                    d_out[smem->output_cta_offset + thread_data_out] = neighbour;
                }
                gather += THREADS_NUM;
            }
        }
    }

    __device__
    void interval_warp_gather(vidType &left, vidType &len) {
        while (__any_sync(FULL_MASK, len >= 32)) {

            // vie for control of warp
            if (len >= 32)
                smem->comm[warp_id][0] = lane_id;

            // winner describes adjlist
            if (smem->comm[warp_id][0] == lane_id) {
                smem->comm[warp_id][1] = left;
                smem->comm[warp_id][2] = left + len / 32 * 32;
                left += len - len % 32;
                len %= 32;
            }

            vidType gather = smem->comm[warp_id][1] + lane_id;
            vidType gather_end = smem->comm[warp_id][2];
            vidType neighbour;
            vidType thread_data_in;
            vidType thread_data_out;
            vidType warp_aggregate;
            while (__any_sync(FULL_MASK, gather < gather_end)) {
                neighbour = gather;
                if (gather < gather_end) {
                    thread_data_in = unvisited(neighbour) ? 1 : 0;
                } else
                    thread_data_in = 0;

                WarpScan(smem->temp_storage[warp_id]).ExclusiveSum(thread_data_in, thread_data_out, warp_aggregate);

                if (0 == lane_id) {
                    smem->output_warp_offset[warp_id] = atomicAdd(d_out_len, warp_aggregate);
                }

                if (thread_data_in) {
                    d_out[smem->output_warp_offset[warp_id] + thread_data_out] = neighbour;
                }
                gather += 32;
            }
        }
    }

    __device__
    void interval_scan_gather(vidType &left, vidType &len) {
        vidType thread_data = len;

        vidType rsv_rank;
        vidType total;
        vidType remain;
        vidType cnt = 0;

        __syncthreads();
        BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        vidType cta_progress = 0;

        while (cta_progress < total) {
            remain = total - cta_progress;

            while ((rsv_rank < cta_progress + THREADS_NUM) && (cnt < len)) {
                smem->left[rsv_rank - cta_progress] = left++;
                rsv_rank++;
                cnt++;
            }
            __syncthreads();

            vidType neighbour;

            if (thread_id < min(remain, THREADS_NUM)) {
                neighbour = smem->left[thread_id];
                thread_data = unvisited(neighbour) ? 1 : 0;
            } else
                thread_data = 0;
            __syncthreads();

            vidType scatter;
            vidType block_aggregate;

            BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, scatter, block_aggregate);
            __syncthreads();

            if (0 == thread_id) {
                smem->output_cta_offset = atomicAdd(d_out_len, block_aggregate);
            }
            __syncthreads();

            if (thread_data) {
                d_out[smem->output_cta_offset + scatter] = neighbour;
            }

            cta_progress += THREADS_NUM;
            __syncthreads();
        }
    }

    __device__
    void expand_intervals(vidType left, vidType len) {
        interval_cta_gather(left, len);
        interval_warp_gather(left, len);
        interval_scan_gather(left, len);
    }

    __device__
    void handle_one_interval_segment(vidType node, volatile eidType &global_offset) {
        CgrReader cgrr;
        cgrr.init(node, graph, global_offset);
        IntervalSegmentHelper sh(node, cgrr);
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
            while ((rsv_rank < cta_progress + THREADS_NUM) && (sh.interval_cnt)) {
                smem->left[rsv_rank - cta_progress] = sh.get_interval_left();
                 smem->len[rsv_rank - cta_progress] = sh.get_interval_len();
                rsv_rank++;
            }
            __syncthreads();

            expand_intervals(smem->left[thread_id], smem->len[thread_id]);

            cta_progress += THREADS_NUM;
            __syncthreads();
        }

        global_offset = cgrr.global_offset;
    }

    __device__
    void handle_segmented_intervals(CgrReader &cgrr) {
        // for retrieve global offset for last segment
        vidType last_segment = SIZE_NONE;

        vidType segment_cnt = cgrr.decode_segment_cnt();

        // cta gather
        while (__syncthreads_or(segment_cnt >= THREADS_NUM)) {
            // vie for control of block
            if (segment_cnt >= THREADS_NUM) smem->comm[0][0] = thread_id;
            __syncthreads();

            // winner describes adjlist
            if (smem->comm[0][0] == thread_id) {
                smem->segment_node[0] = cgrr.node;
                smem->segment_offset[0] = cgrr.global_offset;
                segment_cnt -= THREADS_NUM;
                cgrr.global_offset += INTERVAL_SEGMENT_LEN * THREADS_NUM;

                if (segment_cnt == 0) {
                    last_segment = THREADS_NUM - 1;
                }
            }
            __syncthreads();

            vidType node = smem->segment_node[0];
            volatile eidType offset = smem->segment_offset[0] + INTERVAL_SEGMENT_LEN * thread_id;
            handle_one_interval_segment(node, offset);
            if (thread_id == THREADS_NUM - 1) smem->segment_offset[thread_id] = offset;
            __syncthreads();

            if (last_segment != SIZE_NONE) {
                cgrr.global_offset = smem->segment_offset[last_segment];
                last_segment = SIZE_NONE;
            }
        }

        vidType thread_data = segment_cnt;

        vidType rsv_rank;
        vidType total;

        __syncthreads();
        BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        vidType cta_progress = 0;

        while (cta_progress < total) {
            smem->segment_node[thread_id] = SIZE_NONE;
            __syncthreads();

            while (__any_sync(FULL_MASK, (rsv_rank + 32 < cta_progress + THREADS_NUM) && (segment_cnt >= 32))) {
                // vie for control of warp
                if ((rsv_rank + 32 < cta_progress + THREADS_NUM) && (segment_cnt >= 32)) {
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

            while ((rsv_rank < cta_progress + THREADS_NUM) && segment_cnt) {
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

            handle_one_interval_segment(smem->segment_node[thread_id], smem->segment_offset[thread_id]);

            cta_progress += THREADS_NUM;
            __syncthreads();

            if (last_segment != SIZE_NONE) {
                cgrr.global_offset = smem->segment_offset[last_segment];
                last_segment = SIZE_NONE;
            }
        }
    }

    __device__
    void handle_one_residual_segment(vidType node, eidType global_offset) {

        CgrReader cgrr;
        cgrr.init(node, graph, global_offset);
        ResidualSegmentHelper sh(node, cgrr);
        sh.decode_residual_cnt();

        while (__all_sync(FULL_MASK, sh.residual_cnt)) {
            vidType neighbour = sh.get_residual();
            vidType thread_data = unvisited(neighbour) ? 1 : 0;

            vidType scatter;
            vidType warp_aggregate;

            WarpScan(smem->temp_storage[warp_id]).ExclusiveSum(thread_data, scatter, warp_aggregate);

            if (0 == lane_id) {
                smem->output_warp_offset[warp_id] = atomicAdd(d_out_len, warp_aggregate);
            }

            if (thread_data) {
                d_out[smem->output_warp_offset[warp_id] + scatter] = neighbour;
            }
        }

        vidType thread_data = sh.residual_cnt;

        vidType rsv_rank;
        vidType total;
        vidType remain;

        WarpScan(smem->temp_storage[warp_id]).ExclusiveSum(thread_data, rsv_rank, total);

        vidType warp_progress = 0;
        while (warp_progress < total) {
            remain = total - warp_progress;

            while ((rsv_rank < warp_progress + 32) && (sh.residual_cnt)) {
                smem->left[warp_id * 32 + rsv_rank - warp_progress] = sh.get_residual();
                rsv_rank++;
            }

            vidType neighbour;

            if (lane_id < min(remain, 32)) {
                neighbour = smem->left[thread_id];
                thread_data = unvisited(neighbour) ? 1 : 0;
            } else {
                thread_data = 0;
            }

            vidType scatter;
            vidType warp_aggregate;

            WarpScan(smem->temp_storage[warp_id]).ExclusiveSum(thread_data, scatter, warp_aggregate);

            if (0 == lane_id) {
                smem->output_warp_offset[warp_id] = atomicAdd(d_out_len, warp_aggregate);
            }

            if (thread_data) {
                d_out[smem->output_warp_offset[warp_id] + scatter] = neighbour;
            }

            warp_progress += 32;
        }    
    }

    __device__
    void handle_segmented_residuals(CgrReader &cgrr) {

        vidType segment_cnt = cgrr.decode_segment_cnt();

        // cta gather
        while (__syncthreads_or(segment_cnt >= THREADS_NUM)) {
            // vie for control of block
            if (segment_cnt >= THREADS_NUM) smem->comm[0][0] = thread_id;
            __syncthreads();

            // winner describes adjlist
            if (smem->comm[0][0] == thread_id) {
                smem->segment_node[0] = cgrr.node;
                smem->segment_offset[0] = cgrr.global_offset;
                segment_cnt -= THREADS_NUM;
                cgrr.global_offset += RESIDUAL_SEGMENT_LEN * THREADS_NUM;
            }
            __syncthreads();

            vidType node = smem->segment_node[0];
            eidType offset = smem->segment_offset[0] + RESIDUAL_SEGMENT_LEN * thread_id;
            handle_one_residual_segment(node, offset);
        }

        vidType thread_data = segment_cnt;

        vidType rsv_rank;
        vidType total;

        __syncthreads();
        BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        vidType cta_progress = 0;

        while (cta_progress < total) {
            smem->segment_node[thread_id] = SIZE_NONE;
            __syncthreads();

            while (__any_sync(FULL_MASK, (rsv_rank + 32 < cta_progress + THREADS_NUM) && (segment_cnt >= 32))) {
                // vie for control of warp
                if ((rsv_rank + 32 < cta_progress + THREADS_NUM) && (segment_cnt >= 32)) {
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

            while ((rsv_rank < cta_progress + THREADS_NUM) && segment_cnt) {
                smem->segment_offset[rsv_rank - cta_progress] = cgrr.global_offset;
                smem->segment_node[rsv_rank - cta_progress] = cgrr.node;
                rsv_rank++;
                segment_cnt--;
                cgrr.global_offset += RESIDUAL_SEGMENT_LEN;
            }
            __syncthreads();

            handle_one_residual_segment(smem->segment_node[thread_id], smem->segment_offset[thread_id]);

            cta_progress += THREADS_NUM;
            __syncthreads();
        }
    }

    __device__
    void bitmap(vidType &neighbour) {
        if (neighbour == SIZE_NONE)
            return;

        mask_t bit_loc = 1 << (neighbour % MASK_LEN);
        vidType bit_chunk = visited_mask[neighbour / MASK_LEN];
        if (bit_chunk & bit_loc) {
            neighbour = SIZE_NONE;
            return;
        }
        visited_mask[neighbour / MASK_LEN] = bit_chunk + bit_loc;
    }

    __device__
    void update_label_atomic(vidType &neighbour) {
        if (neighbour == SIZE_NONE) return;

        vidType ret = atomicCAS(labels + neighbour, SIZE_NONE, iteration);
        if (ret != SIZE_NONE) {
            neighbour = SIZE_NONE;
        }
    }

    __device__
    void update_label(vidType &neighbour) {
        if (neighbour == SIZE_NONE)
            return;

        if (labels[neighbour] == SIZE_NONE) {
            labels[neighbour] = iteration;
            return;
        }

        neighbour = SIZE_NONE;
    }

    __device__
    bool unvisited(vidType &neighbour) {
        bitmap(neighbour);
        update_label(neighbour);
        return neighbour != SIZE_NONE;
    }

    __device__
    void process() {
        vidType cta_offset = blockDim.x * blockIdx.x;

        while (cta_offset < d_in_len[0]) {

            vidType node;
            eidType row_begin;
            CgrReader cgrr;

            if (cta_offset + thread_id < d_in_len[0]) {
                node = d_in[cta_offset + thread_id];
                row_begin = offsets[node];
                cgrr.init(node, graph, row_begin);
            } else {
                node = SIZE_NONE;
                cgrr.node = node;
                row_begin = 0;
            }

            // handle intervals
            handle_segmented_intervals(cgrr);

            // handle residuals
            handle_segmented_residuals(cgrr);

            cta_offset += blockDim.x * gridDim.x;
        }
    }
};
