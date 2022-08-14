#include "cgr_compressor.hpp"

int main(int argc,char *argv[]) {
  if (argc != 3) {
    printf("incorrect arguments.\n");
    printf("<input_path> <output_path>\n");
    abort();
  }
  Graph g(argv[1]);
  auto compressor = cgr_compressor(&g, 3, 4, INTERVAL_SEGMENT_LEN, RESIDUAL_SEGMENT_LEN);
  compressor.compress();
  compressor.write_cgr(argv[2]);
  printf("CGR generation completed.\n");
  return 0;
}
