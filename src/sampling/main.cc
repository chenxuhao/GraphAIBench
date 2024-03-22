#include "graph.h"
#include <bits/stdc++.h>
using namespace std;

double khop_sample(Graph &g, vector<int>& initial, int steps, int* sample_size, int total_num, int* result, int pdeg=128, int seed=0);

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>"
              << "[num_gpu(1)] [chunk_size(1024)]\n";
    std::cout << "Example: " << argv[0] << " ../inputs/cora/graph\n";
    exit(1);
  }
  Graph g(argv[1], 0 , 1, 0, 0, 1);
  g.print_meta_data();

  double iElaps;
  int sample_num = argc >= 3 ? atoi(argv[2]) : 128;
  int pdeg = argc >= 4 ? atoi(argv[3]) : 128;
  vector<int> initial(sample_num);
  for (int i = 0; i < sample_num; i++) {
    initial[i] = i;
  }
  int steps = 3;
  int sample_size[] = {15, 10, 5};
  int total_num = initial.size();
  int cur_num = total_num;
  for (int i = 0; i < steps; i++) {
    total_num += cur_num * sample_size[i];
    cur_num *= sample_size[i];
  }
  int* result = new int[total_num];
  iElaps = khop_sample(g, initial, steps, sample_size, total_num, result, pdeg);
  if (sample_num <= 4) {
    for (int i = 0; i < sample_num; i++) {
      cout << "Sample " << i << ":" << endl;
      cout << result[i] << endl;
      cur_num = sample_num;
      int begin = cur_num;
      int offset;
      for (int j = 0; j < steps; j++) {
        offset = i * cur_num * sample_size[j] / sample_num;
        for (int k = 0; k < cur_num * sample_size[j] / sample_num; k++) {
          cout << result[begin + offset + k] << " ";
        }
        cout << endl;
        cur_num *= sample_size[j];
        begin += cur_num;
      }
    }
  }
  cout << "Time elapsed " << iElaps << " sec\n\n";
  delete[] result;

  return 0;
}

