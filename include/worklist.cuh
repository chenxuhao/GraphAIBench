#pragma once
#include "cutil_subset.h"
static int zero = 0;

template <typename TD, typename TI>
class Worklist {
protected:
  TD *d_queue, *h_queue;
  TI *d_size, *d_index;

public:
  Worklist(size_t max_size) {
    h_queue = (TD*) calloc(max_size, sizeof(TD));
    CUDA_SAFE_CALL(cudaMalloc(&d_queue, max_size * sizeof(TD)));
    CUDA_SAFE_CALL(cudaMalloc(&d_size, sizeof(TI)));
    CUDA_SAFE_CALL(cudaMalloc(&d_index, sizeof(TI)));
    CUDA_SAFE_CALL(cudaMemcpy(d_size, &max_size, sizeof(TI), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void *)d_index, &zero, sizeof(zero), cudaMemcpyHostToDevice));
  }
  ~Worklist() {}
  __device__ __host__ void invalidate(int id) { d_queue[id] = TD(-1); }
  void display_items() {
    auto nsize = nitems();
    CUDA_SAFE_CALL(cudaMemcpy(h_queue, d_queue, nsize  * sizeof(TD), cudaMemcpyDeviceToHost));
    printf("Queue: ");
    for(auto i = 0; i < nsize; i++)
      printf("%d %d, ", i, h_queue[i]);
    printf("\n");
    return;
  }
  void reset() {
    CUDA_SAFE_CALL(cudaMemcpy((void *)d_index, &zero, sizeof(TI), cudaMemcpyHostToDevice));
  }
  TI nitems() {
    TI index;
    CUDA_SAFE_CALL(cudaMemcpy(&index, (void *)d_index, sizeof(TI), cudaMemcpyDeviceToHost));
    return index;
  }
  void set_index(TI index) {
    CUDA_SAFE_CALL(cudaMemcpy((void *)d_index, &index, sizeof(TI), cudaMemcpyHostToDevice));
  }
  __device__ int push(TD item) {
    auto lindex = atomicAdd((TI *) d_index, 1);
    if(lindex >= *d_size) return 0;
    d_queue[lindex] = item;
    return 1;
  }
  __device__ int pop(TD &item) {
    auto lindex = atomicSub((TI *) d_index, 1);
    if(lindex <= 0) {
      *d_index = 0;
      return 0;
    }
    item = d_queue[lindex - 1];
    return 1;
  }
};

template <typename TD = vidType, typename TI = vidType>
class Worklist2: public Worklist<TD, TI> {
public:
  Worklist2(TI nsize) : Worklist<TD, TI>(nsize) {}

  template <typename T>
  __device__ __forceinline__ TI push_1item(TI nitem, TD item) {
    assert(nitem == 0 || nitem == 1);
    __shared__ typename T::TempStorage temp_storage;
    __shared__ TI queue_index;
    TI total_items = 0;
    TI thread_data = nitem;
    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);
    __syncthreads();
    if(threadIdx.x == 0) {
      queue_index = atomicAdd((TI*) this->d_index, total_items);
    }
    __syncthreads();
    if(nitem == 1) {
      if(queue_index + thread_data >= *(this->d_size)) {
        printf("GPU: exceeded size: %d %d %d %d %d\n", queue_index, thread_data, *(this->d_size), total_items, *(this->d_index));
        return 0;
      }
      //cub::ThreadStore<cub::STORE_CG>(this->d_queue + queue_index + thread_data, item);
      this->d_queue[queue_index + thread_data] = item;
    }
    __syncthreads();
    return total_items;
  }

  template <typename T>
  __device__ __forceinline__ TI push_nitems(TI n_items, TD *items) {
    __shared__ typename T::TempStorage temp_storage;
    __shared__ TI queue_index;
    TI total_items;
    TI thread_data = n_items;
    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);
    if(threadIdx.x == 0) {	
      queue_index = atomicAdd((TI *) this->d_index, total_items);
      //printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
    }
    __syncthreads();
    for (auto i = 0; i < n_items; i++) {
      //printf("pushing %d to %d\n", items[i], queue_index + thread_data + i);
      if(queue_index + thread_data + i >= *(this->d_size)) {
        printf("GPU: exceeded size: %d %d %d %d\n", queue_index, thread_data, i, *(this->d_size));
        return 0;
      }
      this->d_queue[queue_index + thread_data + i] = items[i];
    }
    return total_items;
  }

  __device__ int pop_id(TI id, TD &item) {
    if (id < *(this->d_index)) {
      //item = cub::ThreadLoad<cub::LOAD_CG>(this->d_queue + id);
      item = this->d_queue[id];
      return 1;
    }
    return 0;
  }
};
