// Extract bits from a bit stream
// *in   : the input bit stream
// offset: the starting point
// bit: number of bits to extract
__device__ uint32_t extract(const uint32_t *in, size_t offset, size_t bit) {
    int      firstBit                = offset;
    int      lastBit                 = firstBit + bit - 1;
    uint32_t packed                  = in[firstBit / 32];
    int      firstBitInPacked        = firstBit % 32;
    uint32_t packedOverflow          = in[lastBit / 32];
    bool     isOverflowing           = lastBit % 32 < firstBitInPacked;
    int      lastBitInPackedOverflow = !isOverflowing ? -1 : lastBit % 32;
    uint32_t outFromPacked =
        ((packed >> firstBitInPacked) & (0xFFFFFFFF >> (32 - (bit - lastBitInPackedOverflow - 1))));
    uint32_t outFromOverflow = (packedOverflow & (0xFFFFFFFF >> (32 - lastBitInPackedOverflow - 1)))
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
__global__ void decode_bp_block(size_t num,
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
__global__ void decode_bp_warp(size_t num,
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
__global__ void decode_vbyte_block(size_t num,
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
__global__ void decode_vbyte_warp(size_t num,
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

