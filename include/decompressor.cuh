#pragma once 
// Extract bits from a bit stream
// *in   : the input bit stream
// offset: the starting point
// bit: number of bits to extract
template <typename T = uint8_t, int L = 8>
__device__ int extract_bits(const T *in, int firstBit, size_t nbits) {
  return int((in[firstBit / L] >> (firstBit%L)) & (T(-1) >> (L - nbits)));
}

template <typename T = uint8_t, int L = 8>
__device__ uint32_t extract_bytes(const T *in, int offset, size_t nbytes) {
  const T *inbyte = in + offset;
  uint32_t val = static_cast<uint32_t>(*inbyte++);
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

// decompress VByte GPU kernel
// Assuming BLOCK_SIZE == pack_size
// Each thread decodes one element at a time
// num     : the number of elements
// *offsets: the endpoints array
// *in     : the input (compressed) bit stream
// *out    : the output (decompressed) integer array
template <size_t pack_size = BLOCK_SIZE>
__device__ void decode_vbyte_block(size_t num,
                                   const uint32_t *offsets,
                                   const uint32_t *in,
                                   uint32_t *out) {
  uint32_t header_len = 2 * (pack_size/32);
  __shared__ uint32_t min_offsets[pack_size + 1];
  for (vidType i = threadIdx.x; i < num; i += BLOCK_SIZE) {
    auto j = i / pack_size; // j is the pack id
    auto begin = offsets[j];// start of the pack
    // Read the header
    auto num_bytes = extract(in + begin, (i%pack_size) * 2, 2) + 1; // read header
    auto num_bits = num_bytes * 8;
    // Store the number of bits of each element in the shared memory
    min_offsets[0] = 0;
    min_offsets[threadIdx.x + 1] = num_bits;
    __syncthreads();
    // Compute prefix sum to get the positions for extracting elements
    typedef cub::BlockScan<uint32_t, pack_size> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).InclusiveSum(min_offsets[threadIdx.x + 1], min_offsets[threadIdx.x + 1]);
    __syncthreads();
    // Extract elements
    out[i]   = extract(in + begin + header_len, min_offsets[threadIdx.x], num_bits);
  }
}

template <size_t pack_size = WARP_SIZE>
__device__ void decode_vbyte_warp(size_t num,
                                  const uint32_t *offsets,
                                  const uint32_t *in,
                                  uint32_t *out) {
  uint32_t header_len = 2 * (pack_size/32);
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ uint32_t shm_offsets[BLOCK_SIZE + BLOCK_SIZE / WARP_SIZE];
  for (vidType i = thread_lane; i < num; i += WARP_SIZE) {
    auto j = i / pack_size; // j is the pack id
    auto k = i % pack_size; // k is the element id within a pack
    auto begin = offsets[j];// start of the pack
    // Read the header
    auto num_bytes = extract(in + begin, k * 2, 2) + 1;
    auto num_bits = num_bytes * 8;
    // Store the number of bits of each element in the shared memory
    auto start = warp_lane*(pack_size+1);
    auto bit_offsets = &shm_offsets[start];
    bit_offsets[0] = 0;
    bit_offsets[thread_lane + 1] = num_bits;
    __syncthreads();
    // Compute prefix sum to get the positions for extracting elements
    typedef cub::BlockScan<uint32_t, pack_size> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).InclusiveSum(bit_offsets[thread_lane + 1], bit_offsets[thread_lane + 1]);
    __syncthreads();
    // Extract elements
    out[i] = extract(in + begin + header_len, bit_offsets[thread_lane], num_bits);
  }
}

template <int pack_size = WARP_SIZE>
__device__ uint32_t decode_vbyte_warp(const size_t length, const uint32_t *in, uint32_t *out) {
  if (length == 0) {
    return 0;
  }
  const uint8_t *inbyte = reinterpret_cast<const uint8_t *>(in);
  uint32_t nvalue = *in; // number of elements to decompress
  inbyte += 4;

  //const uint32_t *const endout = out + nvalue;
  //const uint8_t *const endbyte = inbyte + length - 1;
  //uint32_t val;

  const int header_len = 2 * (pack_size/32); // number of words
  const int hearder_bytes = header_len * 4;

  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ uint32_t shm_offsets[BLOCK_SIZE + BLOCK_SIZE / WARP_SIZE];
  auto end = ((nvalue - 1) / WARP_SIZE + 1) * WARP_SIZE;
  for (vidType i = thread_lane; i < end; i += WARP_SIZE) {
    auto k = thread_lane % pack_size; // k is the element id within a pack
    // Read the header
    auto num_bytes = 0;
    if (i < nvalue) num_bytes = extract_bits(inbyte, k * 2, 2) + 1;
    inbyte += hearder_bytes;
    // Store the number of bits of each element in the shared memory
    auto start = warp_lane * (pack_size+1);
    auto offsets = &shm_offsets[start];
    offsets[0] = 0;
    offsets[thread_lane + 1] = num_bytes;
    __syncthreads();
    // Compute prefix sum to get the positions for extracting elements
    typedef cub::BlockScan<uint32_t, pack_size> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    uint32_t total_bytes = 0;
    BlockScan(temp_storage).InclusiveSum(offsets[thread_lane + 1], offsets[thread_lane + 1], total_bytes);
    __syncthreads();
    // Extract elements
    out[i] = extract_bytes(inbyte, offsets[thread_lane], num_bytes);
    inbyte += total_bytes;
  }
  return nvalue;
}

