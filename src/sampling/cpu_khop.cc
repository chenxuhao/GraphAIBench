#include "graph.h"
#include <bits/stdc++.h>
using namespace std;

inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

inline int next(Graph &g, int transit, int idx) {
    // return transit * 10 + idx;
    int n = g.out_degree(transit);
    if (n == 0) {
        return transit;
    }
    int chosen = rand() % n;
    return g.N(transit, chosen);
}

double khop_sample(Graph &g, vector<int>& initial, int steps, int* sample_size, int total_num, int* result, int pdeg=128, int seed=0) {
    srand(seed);
    int cur_num = initial.size();
    int prev_begin = 0;
    int begin = cur_num;
    int transit, idx;
    double iStart, iElaps;
    for (int i = 0; i < cur_num; i++) {
        result[i] = initial[i];
    }
    iStart = seconds();
    for (int i = 0; i < steps; i++) {
        cur_num *= sample_size[i];
        for (int j = 0; j < cur_num; j++) {
            idx = j % sample_size[i];
            transit = result[j / sample_size[i] + prev_begin];
            result[begin + j] = next(g, transit, idx);
        }
        prev_begin = begin;
        begin += cur_num;
    }
    iElaps = seconds() - iStart;
    return iElaps;
}