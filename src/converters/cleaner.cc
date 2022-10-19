// Copyright 2022 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

int main(int argc, char *argv[]) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP cleaner (" << num_threads << " threads)\n";
  //SemiOutOfCoreGraph g(argv[1]);
  OutOfCoreGraph g(argv[1]);
  //Graph g(argv[1]);
  g.print_meta_data();
  std::cout << "starting cleaning\n";
  g.sort_and_clean_neighbors();
  g.write_to_file(argv[2]);
  return 0;
} 
