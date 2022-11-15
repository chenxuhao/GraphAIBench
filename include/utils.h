#pragma once
#include "common.h"
namespace utils {

template <typename T>
bool search(const std::vector<T> &vlist, T key){
  return std::find(vlist.begin(), vlist.end(), key) != vlist.end();
}

inline long long unsigned parse_nvshmem_symmetric_size(char *value) {
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

inline void split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

inline std::vector<int> PrefixSum(const std::vector<int> &degrees) {
  std::vector<int> sums(degrees.size() + 1);
  int total = 0;
  for (size_t n=0; n < degrees.size(); n++) {
    sums[n] = total;
    total += degrees[n];
  }
  sums[degrees.size()] = total;
  return sums;
}

// sequential prefix sum
template <typename InTy = unsigned, typename OutTy = unsigned>
inline std::vector<OutTy> prefix_sum(const std::vector<InTy>& in) {
  std::vector<OutTy> prefix(in.size() + 1);
  OutTy total = 0;
  for (size_t i = 0; i < in.size(); i++) {
    prefix[i] = total;
    total += (OutTy)in[i];
  }
  prefix[in.size()] = total;
  return prefix;
}

// Utility function to randomly select k items from [begin, end)
template <typename T = int>
inline T* select_k_items(T k, T begin, T end) {
  auto i = begin;

  // reservoir[] is the output array. Initialize
  // it with first k vertices
  T* reservoir = new T[k];
  for (; i < k; i++)
    reservoir[i] = i;

  // Use a different seed value so that we don't get
  // same result each time we run this program
  srand(time(NULL));

  // Iterate from the (k+1)th element to nth element
  for (; i < end; i++) {
    // Pick a random index from 0 to i.
    auto j = rand() % (i + 1);

    // If the randomly picked index is smaller than k,
    // then replace the element present at the index
    // with new element from stream
    if (j < k)
      reservoir[j] = i;
  }
  return reservoir;
}

// Utility function to find ceiling of r in arr[l..h]
template <typename T = int>
inline T find_ceil(T* arr, T r, T l, T h) {
  T mid;
  while (l < h) {
    mid = l + ((h - l) >> 1); // Same as mid = (l+h)/2
    (r > arr[mid]) ? (l = mid + 1) : (h = mid);
  }
  return (arr[l] >= r) ? l : -1;
}

// Utility function to select one element from n elements given a frequency
// (probability) distribution
// https://www.geeksforgeeks.org/random-number-generator-in-arbitrary-probability-distribution-fashion/
template <typename T = int>
T select_one_item(T n, T* dist) {
  T* offsets = new T[n];
  offsets[0] = dist[0];
  // compute the prefix sum of the distribution
  for (T i = 1; i < n; ++i)
    offsets[i] = offsets[i - 1] + dist[i];
  // offsets[n-1] is sum of all frequencies
  T sum = offsets[n - 1];
  T r   = (rand() % sum) + 1;
  // find which range r falls into, and return the index of the range
  return find_ceil(offsets, r, 0, n - 1);
}

}
