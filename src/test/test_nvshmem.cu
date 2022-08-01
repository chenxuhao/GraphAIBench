#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(stmt)                                \
do {                                                    \
  cudaError_t result = (stmt);                          \
  if (cudaSuccess != result) {                          \
    fprintf(stderr, "[%s:%d] CUDA failed with %s \n",   \
        __FILE__, __LINE__, cudaGetErrorString(result));\
    exit(-1);                                           \
  }                                                     \
} while (0)


__global__ void simple_shift(int *destination) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;

  nvshmem_int_p(destination, mype, peer);
}

int main(void) {
  int ndevices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndevices));
  //std::cout << "GPU_total_mem = " << memsize << "\n";
  std::cout << "There are " << ndevices << " GPUs available\n";
 
  int mype_node, msg;
  cudaStream_t stream;

  nvshmem_init();
  int npes = nvshmem_n_pes();
  int mype = nvshmem_my_pe();
  mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  //mype_node = nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD);
  std::cout << "npes = " << npes << ", mype = " << mype << ", mype_node = " << mype_node << "\n";
  cudaSetDevice(mype_node);
  cudaStreamCreate(&stream);

  int *destination = (int *) nvshmem_malloc(sizeof(int));

  simple_shift<<<1, 1, 0, stream>>>(destination);
  nvshmemx_barrier_all_on_stream(stream);
  cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);
  printf("%d: received message %d\n", nvshmem_my_pe(), msg);

  nvshmem_free(destination);
  nvshmem_finalize();
  return 0;
}

