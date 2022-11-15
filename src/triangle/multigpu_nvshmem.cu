// Copyright (c) 2020 MIT
// Author: Xuhao Chen
#include "dist.h"
#include "utils.h"
#include "graph_gpu.h"
#include "graph_partition.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "bs_warp_vertex_nvshmem.cuh"

void TCSolver(Graph &g, uint64_t &total, int n_partitions, int chunk_size) {
#ifdef USE_MPI
  int rank = 0, size = 1;
  initialize_mpi(rank, size);
  if (rank == 0) print_device_info(0);
#endif
  auto nv = g.V();
  auto md = g.get_max_degree();
  auto infile = g.get_inputfile_prefix()+"-part"+std::to_string(rank);
  int ndevices = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
  if (ndevices < n_partitions) {
    std::cout << "Only " << ndevices << " GPUs available\n";
    exit(1);
  } else ndevices = n_partitions;
  int subgraph_size = (nv-1) / n_partitions + 1;
  if (rank == 0) std::cout << "subgraph size: " << subgraph_size << "\n";

  eidType max_subg_ne = 0;
#ifdef USE_MPI
  g.deallocate();
  Graph subg(infile);
  eidType subg_ne = subg.E();
  MPI_Allreduce(&subg_ne, &max_subg_ne, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) std::cout << "Maximum edge count in the subgraphs: " << max_subg_ne << "\n";
#else
  PartitionedGraph pg(&g, ndevices);
  pg.edgecut_partition1D();
  auto num_subgraphs = pg.get_num_subgraphs();
  assert(num_subgraphs == n_partitions);
  for (int i = 0; i < ndevices; i++) {
    auto subg_ne = pg.get_subgraph(i)->E();
    if (subg_ne > max_subg_ne) 
      max_subg_ne = subg_ne;
  }
#endif

#ifdef USE_MPI
  int local_rank = -1;
  int local_size = 1;
  {
    MPI_Comm local_comm;
    MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CALL(MPI_Comm_size(local_comm, &local_size));
    MPI_CALL(MPI_Comm_free(&local_comm));
  }
  if ( 1 < ndevices && ndevices < local_size ) {
    fprintf(stderr,"ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n", ndevices, local_size);
    MPI_CALL(MPI_Finalize());
    exit(1);
  }
  if (1 == ndevices) {
    // Only 1 device visible, assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
    CUDA_SAFE_CALL(cudaSetDevice(0));
  } else {
    CUDA_SAFE_CALL(cudaSetDevice(local_rank));
  }
  CUDA_SAFE_CALL(cudaFree(0));
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  long long unsigned required_symmetric_heap_size = (nv+1) * sizeof(eidType) + max_subg_ne * sizeof(vidType);
  char * value = getenv("NVSHMEM_SYMMETRIC_SIZE");
  if (value) {
    long long unsigned int size_env = utils::parse_nvshmem_symmetric_size(value);
    if (size_env < required_symmetric_heap_size) {
      fprintf(stderr, "ERROR: Required > Current NVSHMEM_SYMMETRIC_SIZE=%s\n", value);
      MPI_CALL(MPI_Finalize());
      exit(1);
    }
  } else {
    char symmetric_heap_size_str[100];
    sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
    if (rank == 0)
      printf("Setting environment variable NVSHMEM_SYMMETRIC_SIZE = %llu\n", required_symmetric_heap_size);
    setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
  }
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#else 
  nvshmem_init();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CUDA_SAFE_CALL(cudaSetDevice(mype_node));
  //cudaStream_t stream;
  //cudaStreamCreate(&stream);
#endif

  int npes = nvshmem_n_pes();
  int mype = nvshmem_my_pe();
#ifdef USE_MPI
  assert(mype == rank);
#else
  auto &subg = *pg.get_subgraph(mype);
#endif
  nvshmem_barrier_all();
  //std::cout << "npes = " << npes << ", mype = " << mype << "\n";

  Timer t;
  t.Start();
  GraphGPU d_graph(nv, max_subg_ne, 0, 0, 0, 0, ndevices, 1);
  d_graph.allocate_nvshmem(nv, max_subg_ne, md, mype);
  std::cout << "PE[" << mype << "] subgraph[" << mype << "] has " << subg.V() << " vertices and " << subg.E() << " edges\n";
  d_graph.init_nvshmem(subg, mype);
  t.Stop();
  std::cout << "PE[" << mype << "] Total time allocating nvshmem and copying subgraphs to GPUs: " << t.Seconds() <<  " sec\n";
 
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = 65536;
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  int max_blocks_per_SM = maximum_residency(warp_vertex_nvshmem, nthreads, 0);
  //std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(6*max_blocks, nblocks); 
  std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";

  size_t nwarps = WARPS_PER_BLOCK;
  size_t per_block_buffer_size = nwarps * size_t(md) * sizeof(vidType);
  size_t buffer_size = nblocks * per_block_buffer_size;
  std::cout << "frontier list size: " << float(buffer_size)/float(1024*1024) << " MB\n";
  vidType *buffers; // each warp has (k-3) vertex sets; each set has size of max_degree
  CUDA_SAFE_CALL(cudaMalloc((void **)&buffers, buffer_size));
  nvshmem_barrier_all();

  double start_time = MPI_Wtime();
  t.Start();
  AccType h_count = 0;
  AccType * d_count = (AccType *)nvshmem_malloc(sizeof(AccType));
  CUDA_SAFE_CALL(cudaMemcpy(d_count, &h_count, sizeof(AccType), cudaMemcpyHostToDevice));
  vidType begin = mype * subgraph_size;
  vidType end = (mype+1) * subgraph_size;
  if (end > nv) end = nv;
  std::cout << "PE[" << mype << "] Start kernel: begin " << begin << " end " << end << " \n";
  warp_vertex_nvshmem<<<nblocks, nthreads>>>(begin, end, d_graph, buffers, mype, ndevices, md, d_count);
  CUDA_SAFE_CALL(cudaMemcpy(&h_count, d_count, sizeof(AccType), cudaMemcpyDeviceToHost));
  t.Stop();
  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();
  nvshmem_barrier_all();
  if (rank == 0) std::cout << "Global runtime = " << end_time - start_time << " sec\n";
  std::cout << "runtime[gpu" << mype << "] = " << t.Seconds() <<  " sec\n";
  //std::cout << "num_triangles[gpu" << mype << "] = " << h_count << "\n";
#ifdef USE_MPI
  uint64_t global_count = 0, local_count = h_count;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  //if (rank == 0) std::cout << "Total triangle count = " << global_count << "\n";
  total = global_count;
  MPI_Finalize();
#else 
  nvshmem_finalize();
#endif
}

