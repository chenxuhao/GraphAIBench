// StreamVByte data format:
//   [count] [count * 2-bits per key] [count * vbyte ints]
//   (out)   | (key)->              | (data)->           |
//   4 bytes | (count+3)/4 bytes    | max of count * 4B  |
// each 8-bit key has four 2-bit lengths: 00=1B, 01=2B, 10=3B, 11=4B
// no particular alignment is assumed or guaranteed for any elements

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "vbyte_encoder.hh"

static inline uint8_t _encode_data(uint32_t val,
                                   uint8_t *__restrict__ *dataPtrPtr) {
  uint8_t *dataPtr = *dataPtrPtr;
  uint8_t code;

  if (val < (1 << 8)) { // 1 byte
    *dataPtr = (uint8_t)(val);
    *dataPtrPtr += 1;
    code = 0;
  } else if (val < (1 << 16)) { // 2 bytes
    *(uint16_t *)dataPtr = (uint16_t)(val);
    *dataPtrPtr += 2;
    code = 1;
  } else if (val < (1 << 24)) { // 3 bytes
    *(uint16_t *)dataPtr = (uint16_t)(val);
    *(dataPtr + 2) = (uint8_t)(val >> 16);
    *dataPtrPtr += 3;
    code = 2;
  } else { // 4 bytes
    *(uint32_t *)dataPtr = val;
    *dataPtrPtr += 4;
    code = 3;
  }

  return code;
}

uint32_t vbyte_encoder::encode(uint32_t count, const uint32_t *in, uint32_t *out, bool add_degree) {
  if (add_degree) *(uint32_t *)out = count;      // first 4 bytes is number of ints
  uint8_t *keyPtr = (uint8_t *)out;              // keys come immediately after 32-bit count
  if (add_degree) keyPtr += 4;
  uint32_t keyLen = (count + 3) / 4;  // 2-bits rounded to full byte
  uint8_t *dataPtr = keyPtr + keyLen; // variable byte data after all keys
  uint32_t bytesWritten = uint32_t(svb_encode_scalar_d1(in, keyPtr, dataPtr, count) - (uint8_t *)out);
  return 1 + (bytesWritten + 3) / 4;
}

uint8_t * vbyte_encoder::svb_encode_scalar(const uint32_t *in,
                                           uint8_t *__restrict__ keyPtr,
                                           uint8_t *__restrict__ dataPtr, uint32_t count) {
  if (count == 0)
    return dataPtr; // exit immediately if no data

  uint8_t shift = 0; // cycles 0, 2, 4, 6, 0, 2, 4, 6, ...
  uint8_t key = 0;
  for (uint32_t c = 0; c < count; c++) {
    if (shift == 8) {
      shift = 0;
      *keyPtr++ = key;
      key = 0;
    }
    uint32_t val = in[c];
    uint8_t code = _encode_data(val, &dataPtr);
    key = (uint8_t)(key | (code << shift));
    shift = (uint8_t)(shift + 2);
  }

  *keyPtr = key;  // write last key (no increment needed)
  return dataPtr; // pointer to first unused data byte
}

uint8_t *vbyte_encoder::svb_encode_scalar_d1_init(const uint32_t *in,
                                                  uint8_t *__restrict__ keyPtr,
                                                  uint8_t *__restrict__ dataPtr,
                                                  uint32_t count, uint32_t prev) {
  if (count == 0)
    return dataPtr; // exit immediately if no data

  uint8_t shift = 0; // cycles 0, 2, 4, 6, 0, 2, 4, 6, ...
  uint8_t key = 0;
  for (uint32_t c = 0; c < count; c++) {
    if (shift == 8) {
      shift = 0;
      *keyPtr++ = key;
      key = 0;
    }
    uint32_t val = in[c] - prev;
    prev = in[c];
    uint8_t code = _encode_data(val, &dataPtr);
    key = (uint8_t)(key | (code << shift));
    shift = (uint8_t)(shift + 2);
  }

  *keyPtr = key;  // write last key (no increment needed)
  return dataPtr; // pointer to first unused data byte
}

uint8_t *vbyte_encoder::svb_encode_scalar_d1(const uint32_t *in, uint8_t *__restrict__ keyPtr,
                                             uint8_t *__restrict__ dataPtr, uint32_t count) {
  return svb_encode_scalar_d1_init(in, keyPtr, dataPtr, count, 0);
}

