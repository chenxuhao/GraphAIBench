#pragma once 
// Extract bits from a bit stream
// *in   : the input bit stream
// offset: the starting point
// bit: number of bits to extract
template <typename T = uint8_t, int L = 8>
__device__ uint32_t extract_bits(const T *in, int firstBit, size_t nbits) {
  return uint32_t((in[firstBit / L] >> (firstBit%L)) & (T(-1) >> (L - nbits)));
}

template <typename T = uint8_t, int L = 8>
__device__ uint32_t extract_bytes(const T *inbyte, int nbytes) {
  //int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  uint32_t val = static_cast<uint32_t>(*inbyte++);
  //if (warp_lane == 0) printf("\t thread %d in warp %d has %d bytes, val=%u, ptr=%p\n", thread_lane, warp_lane, nbytes, val, inbyte);
  if (nbytes > 1) {
    val |= (static_cast<uint32_t>(*inbyte++) << 8);
    if (nbytes > 2) {
      val |= (static_cast<uint32_t>(*inbyte++) << 16);
      if (nbytes > 3) {
        val |= (static_cast<uint32_t>(*inbyte++) << 24);
      }
    }
  }
  return val;
}

template <typename T = uint32_t, int L = 32>
__device__ uint32_t extract(const T *in, size_t offset, size_t bit) {
    int      firstBit                = offset;
    int      lastBit                 = firstBit + bit - 1;
    T        packed                  = in[firstBit / L];
    int      firstBitInPacked        = firstBit % L;
    T        packedOverflow          = in[lastBit / L];
    bool     isOverflowing           = lastBit % L < firstBitInPacked;
    int      lastBitInPackedOverflow = !isOverflowing ? -1 : lastBit % L;
    uint32_t outFromPacked =
        ((packed >> firstBitInPacked) & (T(-1) >> (L - (bit - lastBitInPackedOverflow - 1))));
    uint32_t outFromOverflow = (packedOverflow & (T(-1) >> (L - lastBitInPackedOverflow - 1)))
                               << (bit - lastBitInPackedOverflow - 1);
    return outFromPacked | outFromOverflow;
}

template <bool delta = true>
__device__ vidType decode_streamvbyte_warp(size_t length, const uint32_t *in, uint32_t *out) {
  assert(length != 0);
  ++in;
  uint32_t count = *in; // number of elements to decompress
  if (count == 0) return 0;
  uint8_t *keyPtr = (uint8_t *)in + 4; // full list of keys is next
  uint32_t keyLen = ((count + 3) / 4); // 2-bits per key (rounded up)
  uint8_t *dataPtr = keyPtr + keyLen;  // data starts at end of keys

  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_rounds = (int(count) - 1) / WARP_SIZE + 1;
  typedef cub::WarpScan<uint32_t> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[WARPS_PER_BLOCK];
  uint32_t base = 0;
  //if (thread_lane == 0) printf("warp %d decoding degree = %u, num_rounds=%d, keyPtr=%p, dataPtr=%p\n", warp_lane, count, num_rounds, keyPtr, dataPtr);

  for (auto i = thread_lane; i < num_rounds*WARP_SIZE; i += WARP_SIZE) {
    // Read the header
    uint32_t num_bytes = (i < count) ? (extract_bits(keyPtr, thread_lane * 2, 2) + 1) : 0;
    keyPtr += 8; // move 8 bytes forward; 8*8 = 64 = 32*2
    // Compute prefix sum to get the positions for extracting data elements
    uint32_t offset = 0;
    uint32_t total_bytes = 0;
    WarpScan(temp_storage[warp_lane]).ExclusiveSum(num_bytes, offset, total_bytes);
    //if (warp_lane == 0) printf("thread %d in warp %d has %d bytes, offset=%d, total_bytes=%d\n", thread_lane, warp_lane, num_bytes, offset, total_bytes);
    // Extract elements
    uint32_t val = (i < count) ? extract_bytes(&dataPtr[offset], num_bytes) : 0;
    dataPtr += total_bytes;
    if constexpr (delta) {
      uint32_t delta_val = 0;
      if (thread_lane == 0) val += base;
      // Compute prefix sum for differential/delta coding
      WarpScan(temp_storage[warp_lane]).InclusiveSum(val, delta_val);
      if (i < count) out[i] = delta_val;
      base = __shfl_sync(FULL_MASK, delta_val, WARP_SIZE-1);
    } else {
      if (i < count) out[i] = val;
    }
  }
  return vidType(count);
}

// decompress VByte GPU kernel
// Assuming BLOCK_SIZE == pack_size
// Each thread decodes one element at a time
// length  : the number of compressed words
// *in     : the input (compressed) bit stream
// *out    : the output (decompressed) integer array
template <bool delta = true, size_t pack_size = BLOCK_SIZE>
__device__ vidType decode_varintgb_block(const size_t length, const uint32_t *in, uint32_t *out) {
}

template <bool delta = true, int pack_size = WARP_SIZE>
__device__ vidType decode_varintgb_warp(const size_t length, const uint32_t *in, uint32_t *out) {
  assert(length != 0);
  assert(pack_size >= 4);
  const uint8_t *inbyte = reinterpret_cast<const uint8_t *>(in);
  uint32_t nvalue = *in; // number of elements to decompress
  if (nvalue == 0) return 0;
  inbyte += 4;
  const int header_bytes = (2 * pack_size) / 8;
  assert(header_bytes > 0);
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  int num_rounds = (int(nvalue) - 1) / pack_size + 1;
  typedef cub::WarpScan<uint32_t> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[WARPS_PER_BLOCK];
  vidType base = 0;

  for (int i = 0; i < num_rounds; i ++) {
    __syncwarp();
    int j = thread_lane + i*pack_size; // % pack_size; // k is the element id within a pack
    // Read the header
    uint32_t num_bytes = 0;
    if (thread_lane < pack_size && j < nvalue) num_bytes = extract_bits(inbyte, thread_lane * 2, 2) + 1;
    inbyte += header_bytes;
    // Compute prefix sum to get the positions for extracting elements
    uint32_t offset = 0;
    uint32_t total_bytes = 0;
    WarpScan(temp_storage[warp_lane]).ExclusiveSum(num_bytes, offset, total_bytes);
    // Extract elements
    uint32_t val = 0, delta_val = 0;
    if (thread_lane < pack_size && j < nvalue) val = extract_bytes(&inbyte[offset], num_bytes);
    if (delta && thread_lane == 0) val += base;
    if (delta) WarpScan(temp_storage[warp_lane]).InclusiveSum(val, delta_val);
    if (thread_lane < pack_size && j < nvalue) out[j] = delta ? delta_val : val;
    inbyte += total_bytes;
    base = __shfl_sync(FULL_MASK, delta_val, pack_size-1);
  }
  return vidType(nvalue);
}

// Decompress Binary Packing GPU kernel
// Assuming BLOCK_SIZE >= pack_size
// Each thread decodes one element at a time
// num     : the number of elements
// *offsets: the endpoints array
// *in     : the input (compressed) bit stream
// *out    : the output (decompressed) integer array
template <size_t pack_size = 32>
__device__ void decode_bp_block(size_t num,
                                const uint32_t *offsets,
                                const uint32_t *in,
                                uint32_t *out) {
  for (vidType i = threadIdx.x; i < num; i += BLOCK_SIZE) {
    auto j = i / pack_size; // j is the pack id
    auto begin = offsets[j];// start of the pack
    auto end = offsets[j+1];// end of the pack
    uint8_t nbits = (end - begin)*32 / pack_size; // number of bits per element
    out[i] = extract(in + begin, (i%pack_size) * nbits, nbits);
  }
}

template <size_t pack_size = 32>
__device__ void decode_bp_warp(size_t num,
                               const uint32_t *offsets,
                               const uint32_t *in,
                               uint32_t *out) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  for (vidType i = thread_lane; i < num; i += WARP_SIZE) {
    auto j = i / pack_size; // j is the pack id
    auto begin = offsets[j];// start of the pack
    auto end = offsets[j+1];// end of the pack
    uint8_t nbits = (end - begin)*32 / pack_size; // number of bits per element
    out[i] = extract(in + begin, (i%pack_size) * nbits, nbits);
  }
}

