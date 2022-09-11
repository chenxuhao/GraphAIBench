#pragma once
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <set>
#include <map>
#include <deque>
#include <vector>
#include <limits>
#include <cstdio>
#include <string>
#include <bitset>
#include <cstring>
#include <fstream>
#include <sstream>
#include <climits>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

typedef float   score_t;   // score type for PageRank
typedef float   latent_t;  // latent type for CF
typedef float   feat_t;    // vertex feature type
typedef uint8_t patid_t;   // pattern id type
typedef uint8_t mask_t;    // mask type
typedef uint8_t label_t;   // label type
typedef uint8_t vlabel_t;  // vertex label type
typedef int32_t elabel_t;  // edge label type
//typedef uint16_t elabel_t; // edge label type
//typedef float   elabel_t;  // edge label type
typedef uint8_t cmap_vt;   // cmap value type
typedef uint32_t vidType;   // vertex ID type
typedef int64_t eidType;   // edge ID type
typedef uint32_t IndexT;
typedef int32_t comp_t;    // component ID type for CC
typedef uint64_t emb_index_t; // embedding index type
typedef unsigned long long AccType;

typedef std::vector<patid_t> PidList;    // pattern ID list
typedef std::vector<vidType> VertexList; // vertex ID list
typedef std::vector<std::vector<vidType>> VertexLists;
typedef std::unordered_map<vlabel_t, int> nlf_map;

#define ADJ_SIZE_THREASHOLD 1024
#define FULL_MASK 0xffffffff
#define MAX_PATTERN_SIZE 8
#define MAX_FSM_PATTERN_SIZE 5
#define NUM_BUCKETS 128
#define BUCKET_SIZE 1024
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define BLOCK_SIZE    256
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define MAX_THREADS (30 * 1024)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define MAX_BLOCKS (MAX_THREADS / BLOCK_SIZE)
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define BYTESTOMB(memory_cost) ((memory_cost)/(double)(1024 * 1024))

// PageRank parameters
#define EPSILON 0.0001
#define MAX_ITER 100
const float kDamp = 0.85;
const float epsilon = 0.0000001;
const float epsilon2 = 0.001;

// BFS & SSSP parameters
#define MYINFINITY	1000000000
#define kDistInf UINT_MAX/2

// CF parameters
const int K = 20;        // dimension of the latent vector (number of features)
extern float cf_epsilon; // convergence condition
extern score_t lambda;   // regularization_factor
extern score_t step;     // learning rate in the algorithm
extern int max_iters;    // maximum number of iterations

// Compression
#define USE_INTERVAL 1
#define ZETA_K 3
#define MIN_ITV_LEN 4
#define INTERVAL_SEGMENT_LEN 256
#define RESIDUAL_SEGMENT_LEN 256
static const uint32_t SIZE_NONE = 0xffffffff;
enum OPS {
  DECOMPRESS,
  SETOPS,
};
extern std::map<OPS,double> timers;


enum Status {
  Idle,
  Extending,
  IteratingEdge,
  Working,
  ReWorking
};

#define OP_INTERSECT 'i'
#define OP_DIFFERENCE 'd'
enum SetOp {
  SET_INTERSECTION,
  SET_DIFFERENCE
};

extern std::map<char,double> time_ops;

const std::string long_separator = "--------------------------------------------------------------------\n";
const std::string short_separator = "-----------------------\n";
