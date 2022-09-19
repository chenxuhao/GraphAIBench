#pragma once
#include "common.h"
#include <cub/cub.cuh>
#include <cub/device/dispatch/dispatch_segmented_sort.cuh>

template <typename KeyT = vidType, typename ValueT = cub::NullType, typename OffsetT = vidType>
inline __device__ void cta_sort(vidType num_items, vidType *keys_in, vidType *keys_out) {
  if (num_items <= 0) return;
  /*
  __shared__ vidType neighbors[256];
  assert(num_items <= 256);
  if (threadIdx.x == 0) {
    for (vidType i = num_items; i < 256; i++) {
      neighbors[i] = INT_MAX;
    }
  }
  __syncthreads();
  typedef cub::BlockRadixSort<vidType, BLOCK_SIZE, 1> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  BlockRadixSort(temp_storage).Sort(neighbors+threadIdx.x);
  if (threadIdx.x < num_items) keys_out[threadIdx.x] = neighbors[threadIdx.x];
  */

  typedef cub::DeviceSegmentedSortPolicy<KeyT, ValueT> DispatchSegmentedSort;
  using MaxPolicyT = typename DispatchSegmentedSort::MaxPolicy;
  using ActivePolicyT = typename MaxPolicyT::ActivePolicy;
  using LargeSegmentPolicyT = typename ActivePolicyT::LargeSegmentPolicy;
  using MediumPolicyT = typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT::MediumPolicyT;

  constexpr int radix_bits = LargeSegmentPolicyT::RADIX_BITS;
  const int byte_size  = 8;
  const int num_bits   = sizeof(KeyT) * byte_size;
  const int num_passes = DivideAndRoundUp(num_bits, radix_bits);
  const bool is_num_passes_odd = num_passes & 1;
  bool is_overwrite_okay = false;
  //bool is_num_passes_odd = true;
  cub::DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(keys_in), keys_out);
  cub::detail::temporary_storage::layout<5> temporary_storage_layout;
  auto keys_slot = temporary_storage_layout.get_slot(0);
  //auto large_and_medium_partitioning_slot = temporary_storage_layout.get_slot(2);
  //auto small_partitioning_slot = temporary_storage_layout.get_slot(3);
  //auto group_sizes_slot = temporary_storage_layout.get_slot(4);
  auto keys_allocation = keys_slot->create_alias<KeyT>();
  if (!is_overwrite_okay) keys_allocation.grow(num_items);
  //cub::detail::device_double_buffer<KeyT> d_keys_double_buffer(NULL, NULL);
  cub::detail::device_double_buffer<KeyT> d_keys_double_buffer(
      (is_overwrite_okay || is_num_passes_odd) ? d_keys.Alternate() : keys_allocation.get(),
      (is_overwrite_okay) ? d_keys.Current() : (is_num_passes_odd) ? keys_allocation.get() : d_keys.Alternate());

  using WarpReduceT = cub::WarpReduce<KeyT>;
  using AgentWarpMergeSortT = cub::AgentSubWarpSort<0, MediumPolicyT, KeyT, ValueT, OffsetT>;
  using AgentSegmentedRadixSortT = cub::AgentSegmentedRadixSort<0, LargeSegmentPolicyT, KeyT, ValueT, OffsetT>;
  __shared__ union
  {
    typename AgentSegmentedRadixSortT::TempStorage block_sort;
    typename WarpReduceT::TempStorage warp_reduce;
    typename AgentWarpMergeSortT::TempStorage medium_warp_sort;
  } temp_storage;
  AgentSegmentedRadixSortT agent(num_items, temp_storage.block_sort);

  constexpr int begin_bit = 0;
  constexpr int end_bit = sizeof(KeyT) * 8;
  constexpr int cacheable_tile_size = LargeSegmentPolicyT::BLOCK_THREADS * LargeSegmentPolicyT::ITEMS_PER_THREAD;
  //if (thread_id == 0) printf("cacheable_tile_size=%d\n", cacheable_tile_size);
  if (num_items <= MediumPolicyT::ITEMS_PER_TILE) {
    // Sort by a single warp
    if (threadIdx.x < MediumPolicyT::WARP_THREADS) {
      AgentWarpMergeSortT(temp_storage.medium_warp_sort).ProcessSegment(num_items, keys_in, keys_out, (ValueT*)NULL, (ValueT*)NULL);
    }
  } else if (num_items < cacheable_tile_size) {
    // Sort by a CTA if data fits into shared memory
    agent.ProcessSinglePass(begin_bit, end_bit, keys_in, NULL, keys_out, NULL);
  } else {
    // Sort by a CTA with multiple reads from global memory
    int current_bit = begin_bit;
    int pass_bits = (cub::min)(int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));
    d_keys_double_buffer = cub::detail::device_double_buffer<KeyT>(
                           d_keys_double_buffer.current(), d_keys_double_buffer.alternate());
    agent.ProcessIterative(current_bit, pass_bits, keys_in, NULL, d_keys_double_buffer.current(), NULL);
    current_bit += pass_bits;
    #pragma unroll 1
    while (current_bit < end_bit) {
      pass_bits = (cub::min)(int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));
      cub::CTA_SYNC();
      agent.ProcessIterative(current_bit, pass_bits, d_keys_double_buffer.current(), 
                             NULL, d_keys_double_buffer.alternate(), NULL);
      d_keys_double_buffer.swap();
      current_bit += pass_bits;
    }
  }
}

