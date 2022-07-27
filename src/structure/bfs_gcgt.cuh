#pragma once

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>

#include "utils.cuh"
#include "bfs_gcgt_cta.cuh"

extern cub::CachingDeviceAllocator g_allocator;

template<int THREADS_NUM>
__global__
void cg_expand_contract_kernel(SIZE_TYPE iteration, SIZE_TYPE src, SIZE_TYPE *d_in_len, SIZE_TYPE *d_in,
        SIZE_TYPE *d_out_len, SIZE_TYPE *d_out, OFFSET_TYPE *offsets, GRAPH_TYPE *graph, MASK_TYPE *visited_mask,
        SIZE_TYPE *labels) {

    typedef BfsGcgtCta<THREADS_NUM> CTA;

    __shared__ typename CTA::SMem smem;

    if (iteration == 1) {
        if (threadIdx.x == 0) {
            d_in_len[0] = 1;
            d_in[0] = src;

            MASK_TYPE bit_loc = 1 << (src % MASK_LEN);
            visited_mask[src / MASK_LEN] = bit_loc;

            labels[src] = 0;
        }
        __syncthreads();
    }

    CTA cta(&smem, iteration, src, d_in_len, d_in, d_out_len, d_out,
        offsets, graph, visited_mask, labels);
    cta.process();
}

__host__
double cg_bfs(SIZE_TYPE src, SIZE_TYPE node_num, OFFSET_TYPE *offsets, GRAPH_TYPE *graph, SIZE_TYPE *results) {

    // node frontier double buffer
    cub::DoubleBuffer<SIZE_TYPE> d_frontiers;
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_frontiers.d_buffers[0], sizeof(SIZE_TYPE) * node_num));
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_frontiers.d_buffers[1], sizeof(SIZE_TYPE) * node_num));

    // node frontier lens
    cub::DoubleBuffer<SIZE_TYPE> d_len;
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_len.d_buffers[0], sizeof(SIZE_TYPE)));
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_len.d_buffers[1], sizeof(SIZE_TYPE)));

    // bitmap
    MASK_TYPE *visited_mask;
    CubDebugExit(
            g_allocator.DeviceAllocate((void** )&visited_mask, sizeof(MASK_TYPE) * ((node_num - 1) / MASK_LEN + 1)));
    CubDebugExit(cudaMemset(visited_mask, 0, sizeof(MASK_TYPE) * ((node_num - 1) / MASK_LEN + 1)));

    // bfs labels
    SIZE_TYPE *d_labels;
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_labels, sizeof(SIZE_TYPE) * node_num));
    CubDebugExit(cudaMemset(d_labels, 0xFF, sizeof(SIZE_TYPE) * node_num));

    int iteration = 1;
    SIZE_TYPE h_out_len[1] = {1};

    __dsync__;
    GpuTimer timer;
    timer.Start();

    while (true) {
        SIZE_TYPE BLOCKS_NUM = min(4096 * 100, h_out_len[0] / THREADS_NUM + 1);

        CubDebugExit(cudaMemset(d_len.Alternate(), 0, sizeof(SIZE_TYPE)));
        cg_expand_contract_kernel<THREADS_NUM> <<<BLOCKS_NUM, THREADS_NUM>>>(iteration, src, d_len.Current(),
                d_frontiers.Current(), d_len.Alternate(), d_frontiers.Alternate(), offsets, graph, visited_mask,
                d_labels);
        CubDebugExit(cudaMemcpy(h_out_len, d_len.Alternate(), sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));

        if (h_out_len[0] == 0)
            break;

        d_len.selector ^= 1;
        d_frontiers.selector ^= 1;
        iteration += 1;
    };

    __dsync__;
    timer.Stop();
    auto bfs_time = timer.Elapsed();

    CubDebugExit(cudaMemcpy(results, d_labels, sizeof(SIZE_TYPE) * node_num, cudaMemcpyDeviceToHost));

    if (d_frontiers.d_buffers[0])
        CubDebugExit(g_allocator.DeviceFree(d_frontiers.d_buffers[0]));
    if (d_frontiers.d_buffers[1])
        CubDebugExit(g_allocator.DeviceFree(d_frontiers.d_buffers[1]));

    if (d_len.d_buffers[0])
        CubDebugExit(g_allocator.DeviceFree(d_len.d_buffers[0]));
    if (d_len.d_buffers[1])
        CubDebugExit(g_allocator.DeviceFree(d_len.d_buffers[1]));

    if (visited_mask)
        CubDebugExit(g_allocator.DeviceFree(visited_mask));

    if (d_labels)
        CubDebugExit(g_allocator.DeviceFree(d_labels));

    return bfs_time;
}
