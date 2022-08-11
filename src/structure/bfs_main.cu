#include "cgr_decompressor.cuh"
#include "bfs_gcgt.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

using hCG = thrust::host_vector<vidType>;
using hOS = thrust::host_vector<eidType>;
using dCG = thrust::device_vector<vidType>;
using dOS = thrust::device_vector<eidType>;

int load_compressed_graph(std::string file_path, hCG &hcg, hOS &hos);

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

int load_compressed_graph(std::string file_path, hCG &hcg, hOS &hos) {

	// load graph
	std::ifstream ifs;
	ifs.open(file_path + ".graph", std::ios::in | std::ios::binary | std::ios::ate);

	if (!ifs.is_open()) {
		std::cout << "open graph file failed!" << std::endl;
		return -1;
	}

	std::streamsize size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);

	std::vector<uint8_t> buffer(size);

	ifs.read((char*) buffer.data(), size);

	hcg.clear();
	GRAPH_TYPE tmp = 0;
	for (size_t i = 0; i < buffer.size(); i++) {
		tmp <<= 8;
		tmp += buffer[i];
		if ((i + 1) % GRAPH_BYTE == 0) {
			hcg.push_back(tmp);
		}
	}

	if (size % GRAPH_BYTE) {
		int rem = size % GRAPH_BYTE;
		while (rem % GRAPH_BYTE)
			tmp <<= 8, rem++;
		hcg.push_back(tmp);
	}

	ifs.close();

	// load offset
	SIZE_TYPE num_node;
	hos.clear();
	hos.push_back(0);
	std::ifstream ifs_offset;
	ifs_offset.open(file_path + ".offset", std::ios::in);
	ifs_offset >> num_node;
	OFFSET_TYPE cur;
	for (auto i = 0; i < num_node; i++) {

		ifs_offset >> cur;
		hos.push_back(cur);
	}
	ifs_offset.close();

	return num_node;
}

