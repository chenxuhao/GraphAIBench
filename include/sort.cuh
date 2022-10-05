#pragma once
#include "common.h"
#include <cub/cub.cuh>
#ifdef USE_DEVICE_SORT
#include <cub/device/dispatch/dispatch_segmented_sort.cuh>

template <int BlockThreads, class KeyT>
struct policy_t {
  constexpr static int BLOCK_THREADS = BlockThreads;
  constexpr static int RADIX_BITS = sizeof(KeyT) > 1 ? 6 : 4;

  using LargeSegmentPolicy =
    cub::AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
    9,
    KeyT,
    cub::BLOCK_LOAD_TRANSPOSE,
    cub::LOAD_DEFAULT,
    cub::RADIX_RANK_MEMOIZE,
    cub::BLOCK_SCAN_WARP_SCANS,
    RADIX_BITS>;

  using SmallAndMediumSegmentedSortPolicyT =
    cub::AgentSmallAndMediumSegmentedSortPolicy<
    BLOCK_THREADS,
    // Small policy
    cub::AgentSubWarpMergeSortPolicy<4, // threads per problem
    5, // items per thread
    cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT,
    cub::CacheLoadModifier::LOAD_DEFAULT>,
    // Medium policy
    cub::AgentSubWarpMergeSortPolicy<32, // threads per problem
    5,  // items per thread
    cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT,
    cub::CacheLoadModifier::LOAD_DEFAULT>>;
};

template <typename KeyT, typename ValueT, typename OffsetT>
inline __device__ bool cta_sort(OffsetT num_items, cub::detail::device_double_buffer<KeyT> &keys) {
  if (num_items <= 0) return false;
  bool is_num_passes_odd = true;
  //typedef cub::DeviceSegmentedSortPolicy<KeyT, ValueT> DispatchSegmentedSort;
  //using MaxPolicyT = typename DispatchSegmentedSort::MaxPolicy;
  //using ActivePolicyT = typename MaxPolicyT::ActivePolicy;
  using policy = policy_t<BLOCK_SIZE, KeyT>;
  using LargePolicyT = typename policy::LargeSegmentPolicy;
  using MediumPolicyT = typename policy::SmallAndMediumSegmentedSortPolicyT::MediumPolicyT;
  constexpr int radix_bits = LargePolicyT::RADIX_BITS;
  constexpr bool is_descending = false;

  using WarpReduceT = cub::WarpReduce<KeyT>;
  using AgentWarpMergeSortT = cub::AgentSubWarpSort<is_descending, MediumPolicyT, KeyT, ValueT, OffsetT>;
  using AgentSegmentedRadixSortT = cub::AgentSegmentedRadixSort<is_descending, LargePolicyT, KeyT, ValueT, OffsetT>;

  __shared__ union
  {
    typename AgentSegmentedRadixSortT::TempStorage block_sort;
    typename WarpReduceT::TempStorage warp_reduce;
    typename AgentWarpMergeSortT::TempStorage medium_warp_sort;
  } temp_storage;
  AgentSegmentedRadixSortT agent(num_items, temp_storage.block_sort);

  constexpr int begin_bit = 0;
  constexpr int end_bit = sizeof(KeyT) * 8;
  constexpr int cacheable_tile_size = LargePolicyT::BLOCK_THREADS * LargePolicyT::ITEMS_PER_THREAD;
  ValueT* value_ptr = nullptr;
  //if (thread_id == 0) printf("cacheable_tile_size=%d\n", cacheable_tile_size);
  if (num_items <= MediumPolicyT::ITEMS_PER_TILE) {
    // Sort by a single warp
    if (threadIdx.x < MediumPolicyT::WARP_THREADS) {
      AgentWarpMergeSortT(temp_storage.medium_warp_sort).ProcessSegment(num_items, keys.current(), keys.alternate(), value_ptr, value_ptr);
      keys.swap();
    }
  } else if (num_items < cacheable_tile_size) {
    // Sort by a CTA if data fits into shared memory
    agent.ProcessSinglePass(begin_bit, end_bit, keys.current(), value_ptr, keys.alternate(), value_ptr);
    keys.swap();
  } else {
    // Sort by a CTA with multiple reads from global memory
    int current_bit = begin_bit;
    int pass_bits = (cub::min)(int{radix_bits}, (end_bit - current_bit));
    agent.ProcessIterative(current_bit, pass_bits, keys.current(), value_ptr, keys.alternate(), value_ptr);
    keys.swap();
    current_bit += pass_bits;
    #pragma unroll 1
    while (current_bit < end_bit) {
      pass_bits = (cub::min)(int{radix_bits}, (end_bit - current_bit));
      cub::CTA_SYNC();
      agent.ProcessIterative(current_bit, pass_bits, keys.current(), value_ptr, keys.alternate(), value_ptr);
      keys.swap();
      is_num_passes_odd = is_num_passes_odd ? false : true;
      current_bit += pass_bits;
    }
  }
  return is_num_passes_odd;
}

template <typename KeyT = vidType, typename ValueT = cub::NullType, typename OffsetT = vidType>
//inline __device__ vidType* cta_sort(OffsetT num_items, KeyT *src, KeyT *buffer, KeyT **result) {
inline __device__ vidType* cta_sort(OffsetT num_items, KeyT *src, KeyT *buffer) {
  __shared__ vidType *result;
  cub::detail::device_double_buffer<KeyT> keys(src, buffer);
  bool is_num_passes_odd = cta_sort<KeyT,ValueT,OffsetT>(num_items, keys);
  //*result = keys.current();
  //return is_num_passes_odd;
  result = keys.current();
  __syncthreads();
  return result;
}
#endif

