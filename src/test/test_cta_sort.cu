#include "graph_gpu.h"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__global__ void sort(vidType num_items, vidType *src, vidType *buffer, vidType **result) {
  //cub::detail::device_double_buffer<vidType> keys(src, buffer);
  //cta_sort<vidType, cub::NullType, vidType>(num_items, keys);
  //*result = keys.current();
  bool is_num_passes_odd = false;
  if (blockIdx.x == 0) {
    //bool is_num_passes_odd = cta_sort(num_items, src, buffer, result);
    *result = cta_sort(num_items, src, buffer);
    if (*result == buffer) is_num_passes_odd = true;
    if (threadIdx.x == 0) printf("is_num_passes_odd? %s\n", is_num_passes_odd?"yes":"no");
  }
}

int main() {
  const vidType num_items = 300;
  //vidType h_a[10] = {8, 2, 0, 3, 6, 1, 5, 4, 7, 9};
  vidType *h_a = new vidType[num_items];
  vidType *h_b = new vidType[num_items];
  for (vidType i = 0; i < num_items; i++) {
    h_a[i] = rand() % num_items;
  }
  vidType *d_a, *d_b;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_a, num_items * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, num_items * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, num_items * sizeof(vidType), cudaMemcpyHostToDevice));
  thrust::device_vector<vidType*> result(1);
  vidType** d_result = thrust::raw_pointer_cast(result.data());
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = 1; //(num_items-1)/nthreads+1;
  std::cout << "CUDA sorting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
  sort<<<nblocks, nthreads>>>(num_items, d_a, d_b, d_result);
  vidType *h_result = result[0];
  CUDA_SAFE_CALL(cudaMemcpy(h_b, h_result, num_items * sizeof(vidType), cudaMemcpyDeviceToHost));
  for (vidType i = 0; i < num_items; i++)
    std::cout << "h_result[" << i << "]=" << h_b[i] << "\n";
  if (thrust::is_sorted(thrust::device, h_result, h_result + num_items)) {
    std::cout << num_items << " ok" << std::endl;
  } else {
    std::cout << num_items << " fail" << std::endl;
  }
  return 0;
}
