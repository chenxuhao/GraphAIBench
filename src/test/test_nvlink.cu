#include <iostream>
#include <omp.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <nvToolsExt.h>
#include "cutil_subset.h"

int main(int argc, char* argv[]) {
  bool p2p_works = true;
  int num_devices = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&num_devices));

  // GPUs
  int gpuid_0 = 0;
  int gpuid_1 = 1;

  // Memory Copy Size
  uint32_t size = pow(2, 26); // 2^26 = 67MB

  // Allocate Memory
  uint32_t* dev_0;
  cudaSetDevice(gpuid_0);
  cudaMalloc((void**)&dev_0, size);

  uint32_t* dev_1;
  cudaSetDevice(gpuid_1);
  cudaMalloc((void**)&dev_1, size);

  //Check for peer access between participating GPUs: 
  int can_access_peer_0_1;
  int can_access_peer_1_0;
  cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, gpuid_1);
  cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_1, gpuid_0);
  printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_0, gpuid_1, can_access_peer_0_1);
  printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_1, gpuid_0, can_access_peer_1_0);

  if (can_access_peer_0_1 && can_access_peer_1_0) {
    // Enable P2P Access
    cudaSetDevice(gpuid_0);
    cudaDeviceEnablePeerAccess(gpuid_1, 0);
    cudaSetDevice(gpuid_1);
    cudaDeviceEnablePeerAccess(gpuid_0, 0);
  }

  // Init Timing Data
  uint32_t repeat = 10;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Init Stream
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // ~~ Start Test ~~
  cudaEventRecord(start, stream);

  //Do a P2P memcpy
  for (int i = 0; i < repeat; ++i) {
    cudaMemcpyAsync(dev_0, dev_1, size, cudaMemcpyDeviceToDevice, stream);
  }

  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  // ~~ End of Test ~~

  // Check Timing & Performance
  float time_ms;
  cudaEventElapsedTime(&time_ms, start, stop);
  double time_s = time_ms / 1e3;

  double gb = size * repeat / (double)1e9;
  double bandwidth = gb / time_s;

  printf("Seconds: %f\n", time_s);
  printf("Unidirectional Bandwidth: %f (GB/s)\n", bandwidth);

  if (can_access_peer_0_1 && can_access_peer_1_0) {
    // Shutdown P2P Settings
    cudaSetDevice(gpuid_0);
    cudaDeviceDisablePeerAccess(gpuid_1);
    cudaSetDevice(gpuid_1);
    cudaDeviceDisablePeerAccess(gpuid_0);
  }

  // Clean Up
  cudaFree(dev_0);
  cudaFree(dev_1);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

/*
#pragma omp parallel num_threads(num_devices) 
  int dev_id = omp_get_thread_num();
  CUDA_SAFE_CALL(cudaSetDevice(dev_id));
  CUDA_SAFE_CALL(cudaFree(0));
  int canAccessPeer = 0;
  for (int i = 0; i < num_devices; i++) {
    if (i == dev_id) continue;
    CUDA_SAFE_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, i));
    if (canAccessPeer) {
      CUDA_SAFE_CALL(cudaDeviceEnablePeerAccess(i, 0));
      std::cerr << "P2P access enabled from " << dev_id << " to " << i << std::endl;
    } else {
      std::cerr << "P2P access not available from " << dev_id << " to " << i << std::endl;
    }
  }
*/
  return 0;
}
