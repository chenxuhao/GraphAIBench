#include "compressed_graph.cuh"
#include "bfs_gcgt.cuh"

int main(int argc,char *argv[]) {
  if (argc != 2) {
    std::cout << "incorrect arguments." << std::endl;
    std::cout << "<input_path>" << std::endl;
    abort();
  }
  std::string input_path(argv[1]);
  hCG hcg;
  hOS hos;

  SIZE_TYPE node_num = load_compressed_graph(input_path, hcg, hos);

  // for lo buf
  hcg.push_back(0);
  hcg.push_back(0);
  hcg.push_back(0);
  hcg.push_back(0);

  printf("%s CGR loaded.", input_path.c_str());

  dCG dcg(hcg);
  dOS dos(hos);
  __dsync__;

  thrust::host_vector<SIZE_TYPE> results(node_num);

  // warm up
  cg_bfs(0, node_num, RAW_PTR(dos), RAW_PTR(dcg), RAW_PTR(results));
  cg_bfs(0, node_num, RAW_PTR(dos), RAW_PTR(dcg), RAW_PTR(results));
  cg_bfs(0, node_num, RAW_PTR(dos), RAW_PTR(dcg), RAW_PTR(results));
  cg_bfs(0, node_num, RAW_PTR(dos), RAW_PTR(dcg), RAW_PTR(results));
  cg_bfs(0, node_num, RAW_PTR(dos), RAW_PTR(dcg), RAW_PTR(results));

  double bfs_time_sum = 0.0;
  int iter_num = 100;
  for (int i = 0; i < iter_num; i++) {
    SIZE_TYPE source = rand() % node_num;

    double cur_time = cg_bfs(source, node_num, RAW_PTR(dos), RAW_PTR(dcg), RAW_PTR(results));
    bfs_time_sum += cur_time;

    SIZE_TYPE unvisited_cnt = thrust::count(results.begin(), results.end(), SIZE_NONE);

    printf("[%d]\tsource_node: %d\trunning_time: %.5lf\tvisited_num: %d\n",
        i, source, cur_time, node_num - unvisited_cnt
        );
  }
  printf("experiment completed, average running time: %.5lf s.\n", bfs_time_sum / iter_num);
  return 0;
}
