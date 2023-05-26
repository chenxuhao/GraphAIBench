#pragma once
#include <string>
#include <cstring>

class vbyte_encoder {
public:
  vbyte_encoder(std::string name) : scheme_name(name) {}
  uint32_t encode(uint32_t count, const uint32_t *in, uint32_t *out, bool add_degree = false);
  std::string get_name() const { return scheme_name; }
protected:
  std::string scheme_name;
  // streamvbyte encode functions
  uint8_t * svb_encode_scalar(const uint32_t *in, uint8_t *__restrict__ keyPtr, uint8_t *__restrict__ dataPtr, uint32_t count);
  uint8_t * svb_encode_scalar_d1_init(const uint32_t *in, uint8_t *__restrict__ keyPtr, uint8_t *__restrict__ dataPtr,  uint32_t count, uint32_t);
  uint8_t * svb_encode_scalar_d1(const uint32_t *in, uint8_t *__restrict__ keyPtr, uint8_t *__restrict__ dataPtr, uint32_t count);
};

