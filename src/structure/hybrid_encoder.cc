#include "hybrid_encoder.hh"
#include "codecfactory.h"
#include "platform_atomics.h"
using namespace SIMDCompressionLib;

void hybrid_encoder::print_stats() {
  std::cout << "Number of v-byte coded vertices: " << num_vcoded << "\n";
  std::cout << "Number of unary coded vertices: "  << num_ucoded << "\n";
}

size_t hybrid_encoder::encode(vidType v, vidType deg, vidType *in) {
  size_t nwords = 0;
  //std::cout << "Encoding vertex v=" << v << " degree=" << deg << "\n";
  if (deg > degree_threshold) { // use VByte encoding
    nwords = encode_vbyte(v, deg, in);
    fetch_and_add(num_vcoded, 1);
  } else { // use unary encoding
    nwords = encode_unary(v, deg, in);
    fetch_and_add(num_ucoded, 1);
  }
  return nwords;
}

size_t hybrid_encoder::encode_unary(vidType v, vidType deg, vidType *in) {
  if (deg == 0) return 0;
  auto &bit_array = bit_arrays[v];
  bit_array.clear();
  int64_t value = int_2_nat(int64_t(in[0]) - int64_t(v));
  append_zeta(bit_array, value);
  //std::cout << "Debug: v = " << v << "\n";
  //std::cout << "hybrid unary encoder: first element is " << in[0] << "\n";
  for (vidType i = 1; i < deg; i++) {
    value = int64_t(in[i]) - int64_t(in[i - 1]) - 1;
    append_zeta(bit_array, value);
    //std::cout << "hybrid unary encoder: " << i+1 << "-th element is " << in[i] << "\n";
  }
  assert(bit_array.size() > 0);
  return (bit_arrays[v].size() - 1)/32 + 1; // number of bits --> number of words
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

