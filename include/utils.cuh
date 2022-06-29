#pragma once
#include "cutil_subset.h"

__device__ __forceinline__ unsigned LaneId() {
  unsigned ret;
  asm("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

