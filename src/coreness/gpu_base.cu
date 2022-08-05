#include <cub/cub.cuh>
#include "timer.h"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

void KCoreSolver(Graph &g, std::vector<int> &coreness, vidType &largest_core, int, int) {
}
