#pragma once
#include "graph.h"
#include "unary_encoder.hh"
#define CHECKPOINT 50000000

typedef OutOfCoreGraph GraphTy;

class Compressor {
  std::string scheme;          // compression scheme: VByte or Binary Packing
  std::string out_prefix;      // output file prefix
  bool use_unary;              // VByte schemes do not use unary encoding
  GraphTy *g;                  // input graph; uncompressed
  unary_encoder *encoder;      // encoder
  vidType degree_threshold;    // degree threshold for hybrid scheme

  std::vector<vidType> osizes; // sizes of each compressed edgelist
  std::vector<eidType> rowptr; // row pointers
  std::vector<vidType> buffer;
  void compute_ptrs();
  void write_ptrs_to_disk();
  void write_compressed_edges_to_disk();

  // statistics
  vidType vbyte_count, unary_count, trivial_count; // number of vertices
  vidType vbyte_adj_count, unary_adj_count; // number of neighbors
  int64_t unary_bytes, vbyte_bytes; // number of bytes
 
public:
  Compressor(std::string sch, std::string pre, bool is_unary, 
             GraphTy *graph, unary_encoder *enc, vidType deg = 32) :
    scheme(sch), out_prefix(pre), use_unary(is_unary),
    g(graph), encoder(enc), degree_threshold(deg) {}
  void compress(bool pre_encode=true);
  void write_compressed_graph();
  void write_degrees();
  void print_stats();
};

