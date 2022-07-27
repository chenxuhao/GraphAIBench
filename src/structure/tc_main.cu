#include "compressed_graph.cuh"
#include "triangle_cta_compressed.cuh"

int main(int argc,char *argv[]) {
  if (argc != 2) {
    std::cout << "incorrect arguments." << std::endl;
    std::cout << "<input_path>" << std::endl;
    abort();
  }
  std::string input_path(argv[1]);
  hCG hcg; // compressed edgelists
  hOS hos; // offsets
  SIZE_TYPE nv = load_compressed_graph(input_path, hcg, hos);
  hcg.push_back(0);
  hcg.push_back(0);
  hcg.push_back(0);
  hcg.push_back(0);
  printf("%s CGR loaded.", input_path.c_str());
  dCG dcg(hcg);
  dOS dos(hos);
  __dsync__;
  double elapsed_time = tc_compressed(nv, RAW_PTR(dos), RAW_PTR(dcg));
  printf("running time: %.5lf s.\n", elapsed_time);
  return 0;
}
