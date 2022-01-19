#pragma once

#include <map>
#include <set>
#include <cmath>
#include <vector>
#include <string>
#include <cstdio>
#include <limits>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <cstdlib>
#include <cstddef>
#include <iomanip>
#include <numeric>
#include <utility>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define SAMPLE_FRONTIER 0
#define DEFAULT_NUM_LAYER 2
#define DEFAULT_SIZE_FRONTIER 3000
#define DEFAULT_SIZE_HID 16
#define DEFAULT_RATE_LEARN 0.02
#define DEFAULT_IS_SIGMOID false
#define EVAL_INTERVAL 50

#define ADAM_LR 0.05
#define ADAM_BETA1 0.9
#define ADAM_BETA2 0.999
#define ADAM_EPSILON 0.00000001  

#define OP_DENSEMM   'a'
#define OP_SPARSEMM  'b'
#define OP_RELU      'c'
#define OP_DROPOUT   'd'
#define OP_LOSS      'e'
#define OP_BIAS      'f'
#define OP_REDUCE    'g'
#define OP_NORM      'h'
#define OP_SCORE     'i'
#define OP_ATTN      'j'
#define OP_TRANSPOSE 'k'
#define OP_SAMPLE    'l'
#define OP_COPY      'm'

// GPU related
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_NUM_CLASSES 128
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define USE_CUSPARSE

enum class net_phase { TRAIN, TEST, VAL };
enum class gnn_arch { GCN, GAT, SAGE };
typedef float float_t;
typedef float t_data;             // use this type to denote all the values (e.g., weights, feature values, ...)
typedef int t_idx;                // use this type to denote all the indices (e.g., index of V, index of E, ...)
typedef std::vector<float> vec_t; // feature vector (1D)
typedef float acc_t;              // accuracy type
typedef uint8_t label_t;          // label is for classification (supervised learning)
typedef uint8_t mask_t;           // mask is used to indicate different uses of labels: train, val, test
typedef uint32_t index_t;         // index type
typedef float edata_t;            // edge data type
typedef float vdata_t;            // vertex data type
extern std::map<char,double> time_ops;

