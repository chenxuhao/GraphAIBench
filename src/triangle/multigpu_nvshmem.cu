// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include <cub/cub.cuh>
#include "graph_gpu.h"
#include "graph_partition.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"
#include "mpi.h"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "bs_warp_vertex_nvshmem.cuh"
#include <thread>

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (MPI_SUCCESS != mpi_status) {                                              \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
            exit( mpi_status );                                                       \
        }                                                                             \
    }

long long unsigned parse_nvshmem_symmetric_size(char *value) {
  long long unsigned units, size;
  assert(value != NULL);
  if (strchr(value, 'G') != NULL) {
    units=1e9;
  } else if (strchr(value, 'M') != NULL) {
    units=1e6;
  } else if (strchr(value, 'K') != NULL) {
    units=1e3;
  } else {
    units=1;
  }
  assert(atof(value) >= 0);
  size = (long long unsigned) atof(value) * units;
  return size;
}

void TCSolver(Graph &g, uint64_t &total, int n_gpus, int chunk_size) {
  int ndevices = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
  //size_t memsize = print_device_info(0);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  size_t mem_graph = size_t(nv+1)*sizeof(eidType) + size_t(2)*size_t(ne)*sizeof(vidType);
  //std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  std::vector<AccType> h_counts(ndevices, 0);
  std::vector<Timer> subt(ndevices);
  if (ndevices < n_gpus) {
    std::cout << "Only " << ndevices << " GPUs available\n";
  } else ndevices = n_gpus;

  PartitionedGraph pg(&g, ndevices);
  pg.edgecut_partition1D();
  auto num_subgraphs = pg.get_num_subgraphs();
  int subgraph_size = (nv-1) / num_subgraphs + 1;

#ifdef USE_MPI
  int rank = 0, size = 1;
  //MPI_CALL(MPI_Init(&argc, &argv));
  MPI_Init(NULL, NULL);
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

  int local_rank = -1;
  int local_size = 1;
  {
    MPI_Comm local_comm;
    MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
          &local_comm));

    MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CALL(MPI_Comm_size(local_comm, &local_size));

    MPI_CALL(MPI_Comm_free(&local_comm));
  }
  if ( 1 < ndevices && ndevices < local_size )
  {
    fprintf(stderr,"ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n", ndevices, local_size);
    MPI_CALL(MPI_Finalize());
    exit(1);
  }
  if ( 1 == ndevices ) {
    // Only 1 device visible, assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
    CUDA_RT_CALL(cudaSetDevice(0));
  } else {
    CUDA_RT_CALL(cudaSetDevice(local_rank));
  }
  CUDA_RT_CALL(cudaFree(0));
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  long long unsigned required_symmetric_heap_size = (nv+1) * sizeof(eidType) + ne * sizeof(vidType) + md * sizeof(vidType);
  printf("Setting environment variable NVSHMEM_SYMMETRIC_SIZE = %llu\n", required_symmetric_heap_size);
  char * value = getenv("NVSHMEM_SYMMETRIC_SIZE");
  if (value) {
    long long unsigned int size_env = parse_nvshmem_symmetric_size(value);
    if (size_env < required_symmetric_heap_size) {
      fprintf(stderr, "ERROR: Required > Current NVSHMEM_SYMMETRIC_SIZE=%s\n", value);
      MPI_CALL(MPI_Finalize());
      exit(1);
    }
  } else {
    char symmetric_heap_size_str[100];
    sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
    setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
  }
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#else 
  nvshmem_init();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  std::cout << "mype_node = " << mype_node << "\n";
  cudaSetDevice(mype_node);
  //cudaStream_t stream;
  //cudaStreamCreate(&stream);
#endif

  int npes = nvshmem_n_pes();
  int mype = nvshmem_my_pe();
  //nvshmem_barrier_all();
  std::cout << "npes = " << npes << ", mype = " << mype << "\n";

  eidType max_subg_ne = 0;
  for (int i = 0; i < ndevices; i++) {
    auto subg_ne = pg.get_subgraph(i)->E();
    if (subg_ne > max_subg_ne) 
      max_subg_ne = subg_ne;
  }
  Timer t;
  t.Start();
  GraphGPU d_graph(nv, max_subg_ne, 0, 0, 0, 0, ndevices, 1);
  d_graph.allocate_nvshmem(nv, max_subg_ne, md);
  //for (int i = 0; i < ndevices; i++)
  d_graph.init_nvshmem(*pg.get_subgraph(mype), mype);
  t.Stop();
  std::cout << "Total time allocating nvshmem and copying subgraphs to GPUs: " << t.Seconds() <<  " sec\n";
 
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(warp_vertex_nvshmem, nthreads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(6*max_blocks, nblocks); 
  std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  //t.Start();
  subt[mype].Start();
  AccType * d_count = (AccType *)nvshmem_malloc(sizeof(AccType));
  CUDA_SAFE_CALL(cudaMemcpy(d_count, &h_counts[mype], sizeof(AccType), cudaMemcpyHostToDevice));
  vidType begin = mype * subgraph_size;
  vidType end = (mype+1) * subgraph_size;
  warp_vertex_nvshmem<<<nblocks, nthreads>>>(begin, end, d_graph, mype, ndevices, d_count);
  CUDA_SAFE_CALL(cudaMemcpy(&h_counts[mype], d_count, sizeof(AccType), cudaMemcpyDeviceToHost));
  subt[mype].Stop();
 
/*
  std::vector<AccType *> d_count(ndevices);
  for (int i = 0; i < ndevices; i++) {
    CUDA_SAFE_CALL(cudaSetDevice(i));
    CUDA_SAFE_CALL(cudaMalloc(&d_count[i], sizeof(AccType)));
  }
  std::vector<std::thread> threads;
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
 
  for (int i = 0; i < ndevices; i++) {
    threads.push_back(std::thread([&,i]() {
    cudaSetDevice(i);
    subt[i].Start();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(d_count[i], &h_counts[i], sizeof(AccType), cudaMemcpyHostToDevice));
    vidType begin = i * subgraph_size;
    vidType end = (i+1) * subgraph_size;
    warp_vertex_nvshmem<<<nblocks, nthreads>>>(begin, end, d_graph, i, ndevices, d_count[i]);
    CUDA_SAFE_CALL(cudaMemcpy(&h_counts[i], d_count[i], sizeof(AccType), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    subt[i].Stop();
    }));
  }
  for (auto &thread: threads) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
//*/
  nvshmem_finalize();
  //t.Stop();
  for (int i = 0; i < ndevices; i++) total += h_counts[i];
  for (int i = 0; i < ndevices; i++)
    std::cout << "runtime[gpu" << i << "] = " << subt[i].Seconds() <<  " sec\n";
  std::cout << "runtime = " << t.Seconds() <<  " sec\n";
  std::cout << "total_num_triangles = " << total << "\n";
}

