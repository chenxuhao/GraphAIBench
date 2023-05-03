#pragma once
#include "graph.h"
#include "unary_encoder.hh"

typedef OutOfCoreGraph GraphTy;

class Compressor {
  std::string scheme;          // compression scheme: VByte or Binary Packing
  std::string out_prefix;      // output file prefix
  bool use_unary;              // VByte schemes do not use unary encoding
  bool word_aligned;           // word alignment
  bool byte_aligned;           // byte alignment
  bool use_permutate;          // permutate bytes in every word or not; only used when word-aligned
  GraphTy *g;                  // input graph; uncompressed
  unary_encoder *encoder;      // encoder
  vidType degree_threshold;    // degree threshold for hybrid scheme

  std::vector<vidType> osizes; // sizes of each compressed edgelist
  std::vector<eidType> rowptr; // row pointers
  std::vector<vidType> buffer;
  void permutate_bytes_by_word(std::vector<unsigned char> &buf);
  void compute_ptrs();
  void write_ptrs_to_disk();
  void write_compressed_edges_to_disk();

  // statistics
  vidType vbyte_count, unary_count, trivial_count; // number of vertices
  vidType vbyte_adj_count, unary_adj_count; // number of neighbors
  int64_t unary_bytes, vbyte_bytes; // number of bytes
 
public:
  Compressor(std::string sch,
             std::string pre,
             bool is_unary, 
             GraphTy *graph,
             unary_encoder *enc,
             bool permutate = false, // permutate bytes in every word; only used when word-aligned
             vidType deg = 32,
             int align = 0) :
    scheme(sch),
    out_prefix(pre),
    use_unary(is_unary),
    word_aligned(false),
    byte_aligned(false),
    use_permutate(permutate),
    g(graph),
    encoder(enc),
    degree_threshold(deg) {
      if (align == 1) byte_aligned = true;
      if (align == 2) word_aligned = true;
      if (use_permutate) assert(word_aligned);
  }
  void compress(bool pre_encode=true);
  void write_compressed_graph();
  void write_degrees();
  void print_stats();
};

