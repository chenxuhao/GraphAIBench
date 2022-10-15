#pragma once

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include "utils.cuh"
#include "bfs_gcgt_cta.cuh"

using OFFSET_TYPE = uint64_t;
using GRAPH_TYPE = uint32_t;
const int GRAPH_BYTE = 4;
const int GRAPH_LEN = 32;

cub::CachingDeviceAllocator g_allocator(true);

#define __dsync__ CubDebugExit(cudaDeviceSynchronize())

extern cub::CachingDeviceAllocator g_allocator;

template<int THREADS_NUM>
__global__ void cg_expand_contract_kernel(vidType iteration, vidType src, 
        vidType *d_in_len, vidType *d_in,
        vidType *d_out_len, vidType *d_out, 
        eidType *offsets, vidType *graph, 
        mask_t *visited_mask, vidType *labels) {
  typedef BfsGcgtCta<THREADS_NUM> CTA;
  __shared__ typename CTA::SMem smem;
  if (iteration == 1) {
    if (threadIdx.x == 0) {
      d_in_len[0] = 1;
      d_in[0] = src;
      mask_t bit_loc = 1 << (src % MASK_LEN);
      visited_mask[src / MASK_LEN] = bit_loc;
      labels[src] = 0;
    }
    __syncthreads();
  }

  CTA cta(&smem, iteration, src, d_in_len, d_in, d_out_len, d_out,
      offsets, graph, visited_mask, labels);
  cta.process();
}

__host__ double cg_bfs(vidType src, vidType node_num, eidType *offsets, vidType *graph, vidType *results) {
    // node frontier double buffer
    cub::DoubleBuffer<vidType> d_frontiers;
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_frontiers.d_buffers[0], sizeof(vidType) * node_num));
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_frontiers.d_buffers[1], sizeof(vidType) * node_num));

    // node frontier lens
    cub::DoubleBuffer<vidType> d_len;
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_len.d_buffers[0], sizeof(vidType)));
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_len.d_buffers[1], sizeof(vidType)));

    // bitmap
    mask_t *visited_mask;
    CubDebugExit(
            g_allocator.DeviceAllocate((void** )&visited_mask, sizeof(mask_t) * ((node_num - 1) / MASK_LEN + 1)));
    CubDebugExit(cudaMemset(visited_mask, 0, sizeof(mask_t) * ((node_num - 1) / MASK_LEN + 1)));

    // bfs labels
    vidType *d_labels;
    CubDebugExit(g_allocator.DeviceAllocate((void** )&d_labels, sizeof(vidType) * node_num));
    CubDebugExit(cudaMemset(d_labels, 0xFF, sizeof(vidType) * node_num));

    int iteration = 1;
    vidType h_out_len[1] = {1};

    __dsync__;
    GpuTimer timer;
    timer.Start();

    while (true) {
        vidType BLOCKS_NUM = min(4096 * 100, h_out_len[0] / THREADS_NUM + 1);

        CubDebugExit(cudaMemset(d_len.Alternate(), 0, sizeof(vidType)));
        cg_expand_contract_kernel<THREADS_NUM> <<<BLOCKS_NUM, THREADS_NUM>>>(iteration, src, d_len.Current(),
                d_frontiers.Current(), d_len.Alternate(), d_frontiers.Alternate(), offsets, graph, visited_mask,
                d_labels);
        CubDebugExit(cudaMemcpy(h_out_len, d_len.Alternate(), sizeof(vidType), cudaMemcpyDeviceToHost));

        if (h_out_len[0] == 0)
            break;

        d_len.selector ^= 1;
        d_frontiers.selector ^= 1;
        iteration += 1;
    };

    __dsync__;
    timer.Stop();
    auto bfs_time = timer.Elapsed();

    CubDebugExit(cudaMemcpy(results, d_labels, sizeof(vidType) * node_num, cudaMemcpyDeviceToHost));

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
