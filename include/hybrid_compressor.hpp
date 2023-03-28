#pragma once
#include "unary_compressor.hpp"

class hybrid_compressor : public unary_compressor {
  vidType degree_threshold; // use different compressor for below and above this threshold
  std::vector<bits> edges_unary;
  std::vector<std::vector<uint32_t>> edges_vbyte;
  std::vector<vidType> osizes; // number of words for each compressed edgelist
  std::string vbyte_scheme;
public:
  explicit hybrid_compressor(GraphTy *graph, 
                             int zeta_k = 2,
                             vidType deg = 32, 
                             std::string scheme = "streamvbyte")
          : unary_compressor(graph, zeta_k), degree_threshold(deg), vbyte_scheme(scheme) {}
  void compress();
  void write_degrees(std::string prefix);
  void write_compressed_colidx(std::string prefix);
  void write_compressed_rowptr(std::string prefix);
  void set_degree_threshold(vidType degree) { degree_threshold = degree; }
  void set_vbyte_scheme(std::string scheme) { vbyte_scheme = scheme; }
protected:
  void encode_vertex(const size_type v);
  void encode_vbyte(vidType v, std::vector<uint32_t>& out);
  void encode_unary(vidType v, bits& out);
}; 
