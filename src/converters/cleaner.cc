// Copyright 2022 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

int main(int argc, char *argv[]) {
  //SemiOutOfCoreGraph g(argv[1]);
  //OutOfCoreGraph g(argv[1]);
  Graph g(argv[1]);
  g.print_meta_data();
  std::cout << "starting cleaning\n";
  g.sort_and_clean_neighbors();
  g.write_to_file(argv[2]);
  return 0;
} 
