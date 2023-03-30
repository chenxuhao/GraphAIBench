#include "hybrid_encoder.hh"
#include "codecfactory.h"
using namespace SIMDCompressionLib;

size_t hybrid_encoder::encode(vidType v, vidType deg, vidType *in) {
  size_t nbytes = 0;
  //std::cout << "Encoding vertex v=" << v << " degree=" << deg << "\n";
  if (deg > degree_threshold) { // use VByte encoding
    auto nwords = encode_vbyte(v, deg, in);
    nbytes = 4 * nwords;
  } else { // use unary encoding
    nbytes = encode_unary(v, deg, in);
  }
  return nbytes;
}

size_t hybrid_encoder::encode_unary(vidType v, vidType deg, vidType *in) {
  if (deg == 0) return 0;
  auto &bit_array = bit_arrays[v];
  bit_array.clear();
  int64_t value = int_2_nat(int64_t(in[0]) - int64_t(v));
  append_zeta(bit_array, value);
  for (vidType i = 1; i < deg; i++) {
    value = int64_t(in[i]) - int64_t(in[i - 1]) - 1;
    append_zeta(bit_array, value);
  }
  assert(bit_array.size() > 0);
  return (bit_arrays[v].size() - 1)/8 + 1; // number of bits --> number of bytes
}

size_t hybrid_encoder::encode_vbyte(vidType v, vidType deg, vidType *in) {
  auto &buffer = word_arrays[v];
  buffer.clear();
  size_t outsize = deg + deg/16 + 2;
  if (buffer.size() < outsize) buffer.resize(outsize);
  outsize = buffer.size();
  shared_ptr<IntegerCODEC> schemeptr = CODECFactory::getFromName(vbyte_scheme);
  if (schemeptr.get() == NULL) exit(1);
  schemeptr->encodeArray(in, deg, buffer.data(), outsize);
  //std::cout << "encode_vbyte: outsize=" << outsize << "\n";
  assert(outsize <= deg + deg/16 + 2);
  buffer.resize(outsize);
  return outsize;
}

