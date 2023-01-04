#pragma once
#include "graph.h"

typedef OutOfCoreGraph GraphTy;

class vbyte_compressor {
  std::string scheme;     // compression scheme: VByte or Binary Packing
  std::string out_prefix; // output file prefix
  GraphTy *g;             // input graph; uncompressed
  FILE *of_graph;         // output file handle for the compressed edgelist
  std::vector<vidType> osizes; // sizes of each compressed edgelist
  std::vector<vidType> buffer;
  int encode_vertex(vidType v);
  void write_ptrs_to_disk();
public:
  vbyte_compressor(std::string sch, std::string pre, GraphTy *graph) : 
    scheme(sch), out_prefix(pre), g(graph) {}
  void compress();
};

