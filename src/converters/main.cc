// Copyright 2022 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "converter.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: %s <input_file> <outpt_prefix>\n", argv[0]);
    printf("Example: %s gr ../galois_inputs/mico.gr ../inputs/mico/graph 1 0 0 0\n", argv[0]);
    exit(1);
  }
  int is_bipartite = 0;
  if (argc > 4) is_bipartite = atoi(argv[4]);
  int write_vlabel = 0, write_elabel = 0;
  if (argc>5) write_vlabel = atoi(argv[5]);
  if (argc>6) write_elabel = atoi(argv[6]);
  //int write_masks = 0;
  //if (argc>7) write_feats = atoi(argv[7]);
  //if (argc>8) write_masks = atoi(argv[8]);
 
  //Converter converter(argv[1], argv[2], is_bipartite);
  //converter.generate_binary_graph(argv[3], 1, 1, write_vlabel, write_elabel);

  Converter converter;
  std::cout << argv[1] << "\n";
  //assert(argv[1] == "gr");
  converter.splitGRFile(argv[2], argv[3]);
  return 0;
}

