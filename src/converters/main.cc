// Copyright 2022 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "converter.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: %s <graph>\n", argv[0]);
    printf("Example: %s gr ../galois_inputs/mico.gr \n", argv[0]);
    exit(1);
  }
  int is_bipartite = 0;
  if (argc > 4) is_bipartite = atoi(argv[4]);
  Converter converter(argv[1], argv[2], is_bipartite);
  converter.generate_binary_graph(argv[3]);
  return 0;
}

