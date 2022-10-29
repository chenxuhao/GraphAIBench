#include "cgr_compressor.hpp"

int main(int argc,char *argv[]) {
  if (argc < 3) {
    printf("incorrect arguments.\n");
    printf("<input_path> <output_path> [use_interval] [add_degree]\n");
    abort();
  }
  //Graph g(argv[1]);
  OutOfCoreGraph g(argv[1]);
  g.print_meta_data();

  int zeta_k = 3, use_interval = 1, add_degree = 0;
  if (argc > 3) zeta_k = atoi(argv[3]);
  if (argc > 4) use_interval = atoi(argv[4]);
  if (argc > 5) add_degree = atoi(argv[5]);
 
  auto compressor = cgr_compressor(&g, zeta_k, MIN_ITV_LEN, INTERVAL_SEGMENT_LEN, RESIDUAL_SEGMENT_LEN);
  compressor.compress(use_interval, add_degree);
  compressor.write_cgr(argv[2]);
  printf("CGR generation completed.\n");
  return 0;
}
