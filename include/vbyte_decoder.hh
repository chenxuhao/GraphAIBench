#pragma once
#include <string>
#include <cstring>

class vbyte_decoder {
 public:
  vbyte_decoder(std::string name) : scheme_name(name) {}
  void decode(uint32_t count, const uint32_t *in, uint32_t *out) {
    if (count == 0) return;
    uint8_t *keyPtr = (uint8_t *)in;
    uint32_t keyLen = ((count + 3) / 4); // 2-bits per key (rounded up)
    uint8_t *dataPtr = keyPtr + keyLen;  // data starts at end of keys
    svb_decode_avx_d1_simple(out, keyPtr, dataPtr, count);
  }
  uint32_t decode(const uint32_t *in, uint32_t *out) {
    //++in;                             // number of encoded bytes
    uint32_t count = *(uint32_t *)in; // next 4 bytes is number of ints
    //nvalue = count;
    if (count == 0) return 0;
    uint8_t *keyPtr = (uint8_t *)in + 4; // full list of keys is next
    uint32_t keyLen = ((count + 3) / 4); // 2-bits per key (rounded up)
    uint8_t *dataPtr = keyPtr + keyLen;  // data starts at end of keys
    svb_decode_avx_d1_simple(out, keyPtr, dataPtr, count);
    return count;
  }
 private:
  std::string scheme_name;
  // streamvbyte decode functions
  uint8_t * svb_decode_avx_d1_init(uint32_t *out, uint8_t *__restrict__ keyPtr, uint8_t *__restrict__ dataPtr, uint64_t count, uint32_t prev);
  uint8_t * svb_decode_scalar_d1_init(uint32_t *outPtr, const uint8_t *keyPtr, uint8_t *dataPtr, uint32_t count, uint32_t prev);
  uint8_t * svb_decode_avx_d1_simple(uint32_t *out, uint8_t *__restrict__ keyPtr, uint8_t *__restrict__ dataPtr, uint64_t count);
};

