#pragma once

template<int THREADS_NUM>
struct BfsGcgtCta {
    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    typedef cub::WarpScan<SIZE_TYPE> WarpScan;

    struct SMem {
        typename BlockScan::TempStorage block_temp_storage;
        typename WarpScan::TempStorage temp_storage[THREADS_NUM / 32];

        volatile SIZE_TYPE segment_node[THREADS_NUM];
        volatile OFFSET_TYPE segment_offset[THREADS_NUM];

        volatile SIZE_TYPE left[THREADS_NUM];
        volatile SIZE_TYPE len[THREADS_NUM];

        volatile SIZE_TYPE comm[THREADS_NUM / 32][32];

        volatile SIZE_TYPE output_cta_offset;
        volatile SIZE_TYPE output_warp_offset[THREADS_NUM / 32];
    };

    SIZE_TYPE iteration;
    SIZE_TYPE src;
    SIZE_TYPE *d_in_len;
    SIZE_TYPE *d_in;
    SIZE_TYPE *d_out_len;
    SIZE_TYPE *d_out;
    OFFSET_TYPE *offsets;
    GRAPH_TYPE *graph;
    MASK_TYPE *visited_mask;
    SIZE_TYPE *labels;
    SMem *smem;

    SIZE_TYPE thread_id;
    SIZE_TYPE lane_id;
    SIZE_TYPE warp_id;

    __device__ BfsGcgtCta(SMem *smem, SIZE_TYPE iteration, SIZE_TYPE src, SIZE_TYPE *d_in_len, SIZE_TYPE *d_in,
            SIZE_TYPE *d_out_len, SIZE_TYPE *d_out, OFFSET_TYPE *offsets, GRAPH_TYPE *graph, MASK_TYPE *visited_mask,
            SIZE_TYPE *labels) :
            smem(smem), iteration(iteration), src(src), d_in_len(d_in_len), d_in(d_in), d_out_len(d_out_len), d_out(
                    d_out), offsets(offsets), graph(graph), visited_mask(visited_mask), labels(labels) {
        thread_id = threadIdx.x;
        lane_id = thread_id % 32;
        warp_id = thread_id / 32;
    }

    __device__
    void interval_cta_gather(SIZE_TYPE &left, SIZE_TYPE &len) {

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

            SIZE_TYPE gather = smem->comm[0][1] + thread_id;
            SIZE_TYPE gather_end = smem->comm[0][2];
            SIZE_TYPE neighbour;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE block_aggregate;
            
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
    void interval_warp_gather(SIZE_TYPE &left, SIZE_TYPE &len) {
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

            SIZE_TYPE gather = smem->comm[warp_id][1] + lane_id;
            SIZE_TYPE gather_end = smem->comm[warp_id][2];
            SIZE_TYPE neighbour;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE warp_aggregate;
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
    void interval_scan_gather(SIZE_TYPE &left, SIZE_TYPE &len) {
        SIZE_TYPE thread_data = len;

        SIZE_TYPE rsv_rank;
        SIZE_TYPE total;
        SIZE_TYPE remain;
        SIZE_TYPE cnt = 0;

        __syncthreads();
        BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        SIZE_TYPE cta_progress = 0;

        while (cta_progress < total) {
            remain = total - cta_progress;

            while ((rsv_rank < cta_progress + THREADS_NUM) && (cnt < len)) {
                smem->left[rsv_rank - cta_progress] = left++;
                rsv_rank++;
                cnt++;
            }
            __syncthreads();

            SIZE_TYPE neighbour;

            if (thread_id < min(remain, THREADS_NUM)) {
                neighbour = smem->left[thread_id];
                thread_data = unvisited(neighbour) ? 1 : 0;
            } else
                thread_data = 0;
            __syncthreads();

            SIZE_TYPE scatter;
            SIZE_TYPE block_aggregate;

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
    void expand_intervals(SIZE_TYPE left, SIZE_TYPE len) {
        interval_cta_gather(left, len);
        interval_warp_gather(left, len);
        interval_scan_gather(left, len);
    }

    __device__
    void handle_one_interval_segment(SIZE_TYPE node, volatile OFFSET_TYPE &global_offset) {
        CgrReader cgrr;
        cgrr.init(node, graph, global_offset);
        IntervalSegmentHelper sh(node, cgrr);
        sh.decode_interval_cnt();

        SIZE_TYPE thread_data = sh.interval_cnt;

        SIZE_TYPE rsv_rank;
        SIZE_TYPE total;

        __syncthreads();
        BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        SIZE_TYPE cta_progress = 0;

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
        SIZE_TYPE last_segment = SIZE_NONE;

        SIZE_TYPE segment_cnt = cgrr.decode_segment_cnt();

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

            SIZE_TYPE node = smem->segment_node[0];
            volatile OFFSET_TYPE offset = smem->segment_offset[0] + INTERVAL_SEGMENT_LEN * thread_id;
            handle_one_interval_segment(node, offset);
            if (thread_id == THREADS_NUM - 1) smem->segment_offset[thread_id] = offset;
            __syncthreads();

            if (last_segment != SIZE_NONE) {
                cgrr.global_offset = smem->segment_offset[last_segment];
                last_segment = SIZE_NONE;
            }
        }

        SIZE_TYPE thread_data = segment_cnt;

        SIZE_TYPE rsv_rank;
        SIZE_TYPE total;

        __syncthreads();
        BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        SIZE_TYPE cta_progress = 0;

        while (cta_progress < total) {
            smem->segment_node[thread_id] = SIZE_NONE;
            __syncthreads();

            while (__any_sync(FULL_MASK, (rsv_rank + 32 < cta_progress + THREADS_NUM) && (segment_cnt >= 32))) {
                // vie for control of warp
                if ((rsv_rank + 32 < cta_progress + THREADS_NUM) && (segment_cnt >= 32)) {
                    smem->comm[warp_id][0] = lane_id;
                }
                SIZE_TYPE boss_lane = smem->comm[warp_id][0];
                SIZE_TYPE boss_node = __shfl_sync(FULL_MASK, cgrr.node, boss_lane);
                OFFSET_TYPE boss_global_offset = __shfl_sync(FULL_MASK, cgrr.global_offset, boss_lane);
                SIZE_TYPE boss_rsv_rank = __shfl_sync(FULL_MASK, rsv_rank, boss_lane);

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
    void handle_one_residual_segment(SIZE_TYPE node, OFFSET_TYPE global_offset) {

        CgrReader cgrr;
        cgrr.init(node, graph, global_offset);
        ResidualSegmentHelper sh(node, cgrr);
        sh.decode_residual_cnt();

        while (__all_sync(FULL_MASK, sh.residual_cnt)) {
            SIZE_TYPE neighbour = sh.get_residual();
            SIZE_TYPE thread_data = unvisited(neighbour) ? 1 : 0;

            SIZE_TYPE scatter;
            SIZE_TYPE warp_aggregate;

            WarpScan(smem->temp_storage[warp_id]).ExclusiveSum(thread_data, scatter, warp_aggregate);

            if (0 == lane_id) {
                smem->output_warp_offset[warp_id] = atomicAdd(d_out_len, warp_aggregate);
            }

            if (thread_data) {
                d_out[smem->output_warp_offset[warp_id] + scatter] = neighbour;
            }
        }

        SIZE_TYPE thread_data = sh.residual_cnt;

        SIZE_TYPE rsv_rank;
        SIZE_TYPE total;
        SIZE_TYPE remain;

        WarpScan(smem->temp_storage[warp_id]).ExclusiveSum(thread_data, rsv_rank, total);

        SIZE_TYPE warp_progress = 0;
        while (warp_progress < total) {
            remain = total - warp_progress;

            while ((rsv_rank < warp_progress + 32) && (sh.residual_cnt)) {
                smem->left[warp_id * 32 + rsv_rank - warp_progress] = sh.get_residual();
                rsv_rank++;
            }

            SIZE_TYPE neighbour;

            if (lane_id < min(remain, 32)) {
                neighbour = smem->left[thread_id];
                thread_data = unvisited(neighbour) ? 1 : 0;
            } else {
                thread_data = 0;
            }

            SIZE_TYPE scatter;
            SIZE_TYPE warp_aggregate;

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

        SIZE_TYPE segment_cnt = cgrr.decode_segment_cnt();

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

            SIZE_TYPE node = smem->segment_node[0];
            OFFSET_TYPE offset = smem->segment_offset[0] + RESIDUAL_SEGMENT_LEN * thread_id;
            handle_one_residual_segment(node, offset);
        }

        SIZE_TYPE thread_data = segment_cnt;

        SIZE_TYPE rsv_rank;
        SIZE_TYPE total;

        __syncthreads();
        BlockScan(smem->block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total);
        __syncthreads();

        SIZE_TYPE cta_progress = 0;

        while (cta_progress < total) {
            smem->segment_node[thread_id] = SIZE_NONE;
            __syncthreads();

            while (__any_sync(FULL_MASK, (rsv_rank + 32 < cta_progress + THREADS_NUM) && (segment_cnt >= 32))) {
                // vie for control of warp
                if ((rsv_rank + 32 < cta_progress + THREADS_NUM) && (segment_cnt >= 32)) {
                    smem->comm[warp_id][0] = lane_id;
                }
                SIZE_TYPE boss_lane = smem->comm[warp_id][0];
                SIZE_TYPE boss_node = __shfl_sync(FULL_MASK, cgrr.node, boss_lane);
                OFFSET_TYPE boss_global_offset = __shfl_sync(FULL_MASK, cgrr.global_offset, boss_lane);
                SIZE_TYPE boss_rsv_rank = __shfl_sync(FULL_MASK, rsv_rank, boss_lane);

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
    void bitmap(SIZE_TYPE &neighbour) {
        if (neighbour == SIZE_NONE)
            return;

        MASK_TYPE bit_loc = 1 << (neighbour % MASK_LEN);
        SIZE_TYPE bit_chunk = visited_mask[neighbour / MASK_LEN];
        if (bit_chunk & bit_loc) {
            neighbour = SIZE_NONE;
            return;
        }
        visited_mask[neighbour / MASK_LEN] = bit_chunk + bit_loc;
    }

    __device__
    void update_label_atomic(SIZE_TYPE &neighbour) {
        if (neighbour == SIZE_NONE) return;

        SIZE_TYPE ret = atomicCAS(labels + neighbour, SIZE_NONE, iteration);
        if (ret != SIZE_NONE) {
            neighbour = SIZE_NONE;
        }
    }

    __device__
    void update_label(SIZE_TYPE &neighbour) {
        if (neighbour == SIZE_NONE)
            return;

        if (labels[neighbour] == SIZE_NONE) {
            labels[neighbour] = iteration;
            return;
        }

        neighbour = SIZE_NONE;
    }

    __device__
    bool unvisited(SIZE_TYPE &neighbour) {
        bitmap(neighbour);
        update_label(neighbour);
        return neighbour != SIZE_NONE;
    }

    __device__
    void process() {
        SIZE_TYPE cta_offset = blockDim.x * blockIdx.x;

        while (cta_offset < d_in_len[0]) {

            SIZE_TYPE node;
            OFFSET_TYPE row_begin;
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