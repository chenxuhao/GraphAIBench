#pragma once
#include "graph.h"
#include "cgr_encoder.hh"

typedef OutOfCoreGraph GraphTy;

class Compressor {
  std::string scheme;          // compression scheme: VByte or Binary Packing
  std::string out_prefix;      // output file prefix
  bool use_unary;              // VByte schemes do not use unary encoding
  GraphTy *g;                  // input graph; uncompressed
  unary_encoder *encoder;      // encoder
  std::vector<vidType> osizes; // sizes of each compressed edgelist
  std::vector<eidType> rowptr; // row pointers
  std::vector<vidType> buffer;
  void compute_ptrs();
  void write_ptrs_to_disk();
  void write_compressed_edges_to_disk();

public:
  Compressor(std::string sch, std::string pre, bool is_unary, GraphTy *graph, unary_encoder *enc) :
    scheme(sch), out_prefix(pre), use_unary(is_unary), g(graph), encoder(enc) {}
  void compress(bool pre_encode=true);
  void write_compressed_graph();
};

