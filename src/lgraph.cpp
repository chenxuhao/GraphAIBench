// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "lgraph.h"
#include "reader.h"

void LearningGraph::compute_edge_data() {
  if (edge_data_ && edata_size < num_edges_) delete[] edge_data_; // graph size may change due to subgraph sampling
  if (edge_data_ == NULL) edge_data_ = new edata_t[num_edges_];
  edata_size = num_edges_;
  #pragma omp parallel for
  for (size_t i = 0; i < num_vertices_; i ++) {
    float c_i = std::sqrt(float(get_degree(i)));
    for (auto e = edge_begin(i); e != edge_end(i); e++) {
      const auto j = getEdgeDst(e);
      float c_j  = std::sqrt(float(get_degree(j)));
      if (c_i == 0.0 || c_j == 0.0) edge_data_[e] = 0.0;
      else edge_data_[e] = 1.0 / (c_i * c_j);
    }
  }
}

void LearningGraph::compute_vertex_data() {
  //std::cout << "Computing vertex data\n";
  if (vertex_data_ && vdata_size < num_vertices_) delete[] vertex_data_; // graph size may change due to subgraph sampling
  if (vertex_data_ == NULL) vertex_data_ = new vdata_t[num_vertices_];
  vdata_size = num_vertices_;
  #pragma omp parallel for
  for (size_t v = 0; v < num_vertices_; v ++) {
    auto degree = get_degree(v);
    float temp = std::sqrt(float_t(degree));
    if (temp == 0.0) vertex_data_[v] = 0.0;
    else vertex_data_[v] = 1.0 / temp;
  }
}

void LearningGraph::dealloc() {
  delete[] rowptr_;
  delete[] colidx_;
  if (vertex_data_) delete[] vertex_data_;
  if (edge_data_) delete[] edge_data_;
}

void LearningGraph::update_feat_len(size_t len) {
  for (int i = 0; i < num_subgraphs; ++i) {
    #pragma omp parallel for
    for (index_t j = 0; j < nvs_of_subgraphs[i]; ++j) {
      delete[] partial_sums[i][j];
      partial_sums[i][j] = new float[len];
    }
  }
}

// This implements the CSR segmenting technique for graph computation
// This is for pull model, using incomming edges
void LearningGraph::segmenting(size_t len) {
  num_subgraphs = (num_vertices_ - 1) / SUBGRAPH_SIZE + 1;
  num_ranges = (num_vertices_ - 1) / RANGE_WIDTH + 1;
  printf("number of subgraphs and ranges: %d, %d\n", num_subgraphs, num_ranges);

  Timer t;
  t.Start();
/*
  rowptr_blocked.resize(num_subgraphs);
  colidx_blocked.resize(num_subgraphs);
  //edge_data_blocked.resize(num_subgraphs);
  nvs_of_subgraphs.resize(num_subgraphs, 0);
  nes_of_subgraphs.resize(num_subgraphs, 0);
  idx_map.resize(num_subgraphs);
  range_indices.resize(num_subgraphs);
  //partial_sums.resize(num_subgraphs);
  std::vector<int> flag(num_subgraphs, false);
  //for (int i = 0; i < num_subgraphs; ++i) {
  //  nvs_of_subgraphs[i] = 0;
  //  nes_of_subgraphs[i] = 0;
  //}

  std::cout << "calculating number of vertices and edges in each subgraph\n";
  for (index_t dst = 0; dst < num_vertices_; ++ dst) {
    auto start = rowptr_[dst];
    auto end = rowptr_[dst+1];
    for (auto j = start; j < end; ++j) {
      auto src = colidx_[j];
      auto bcol = src / SUBGRAPH_SIZE;
      flag[bcol] = true;
      nes_of_subgraphs[bcol]++;
    }
    for (int i = 0; i < num_subgraphs; ++i) {
      if (flag[i]) nvs_of_subgraphs[i] ++;
    }
    for (int i = 0; i < num_subgraphs; ++i) flag[i] = false;
  }

  std::cout << "allocating memory for each subgraph\n";
  for (int i = 0; i < num_subgraphs; ++i) {
    rowptr_blocked[i].resize(nvs_of_subgraphs[i]+1);
    colidx_blocked[i].resize(nes_of_subgraphs[i]);
    //edge_data_blocked[i].resize(nes_of_subgraphs[i]);
    //partial_sums[i].resize(nvs_of_subgraphs[i]);
    nes_of_subgraphs[i] = 0;
    rowptr_blocked[i][0] = 0;
  }
  for (int i = 0; i < num_subgraphs; ++i) {
    //#pragma omp parallel for
    for (index_t j = 0; j < nvs_of_subgraphs[i]; ++j) {
      //partial_sums[i][j].reserve(len);
      //partial_sums[i][j].resize(len);
    }
  }

  std::cout << "allocating memory for IdxMap, RangeIndices and IntermBuf\n";
  for (int i = 0; i < num_subgraphs; ++i) {
    idx_map[i].resize(nvs_of_subgraphs[i]);
    range_indices[i].resize(num_ranges+1);
    range_indices[i][0] = 0;
  }

  std::cout << "constructing the blocked CSR\n";
  std::vector<int> index(num_subgraphs, 0);
  for (index_t dst = 0; dst < num_vertices_; ++ dst) {
    for (index_t j = rowptr_[dst]; j < rowptr_[dst+1]; ++j) {
      index_t src = colidx_[j];
      int bcol = src / SUBGRAPH_SIZE;
      colidx_blocked[bcol][nes_of_subgraphs[bcol]] = src;
      //assert(!edge_data_.empty());
      //edge_data_blocked[bcol][nes_of_subgraphs[bcol]] = edge_data_[j];
      flag[bcol] = true;
      nes_of_subgraphs[bcol]++;
    }
    for (int i = 0; i < num_subgraphs; ++i) {
      if (flag[i]) {
        idx_map[i][index[i]] = dst;
        rowptr_blocked[i][++index[i]] = nes_of_subgraphs[i];
      }
    }
    for (int i = 0; i < num_subgraphs; ++i) flag[i] = false;
  }
  //printf("printing subgraphs:\n");
  //for (int i = 0; i < num_subgraphs; ++i) {
  //  printf("\tprinting subgraph[%d] (%d vertices, %d edges):\n", i, nvs_of_subgraphs[i], nes_of_subgraphs[i]);
  //}

  std::cout << "constructing IdxMap and RangeIndices\n";
  for (int i = 0; i < num_subgraphs; ++i) {
    std::vector<int> counts(num_ranges, 0);
    for (index_t j = 0; j < nvs_of_subgraphs[i]; ++ j) {
      int dst = idx_map[i][j];
      counts[dst/RANGE_WIDTH] ++;
    }
    for (int j = 1; j < num_ranges+1; ++j) {
      range_indices[i][j] = range_indices[i][j-1] + counts[j-1];
    }
  }
  partitioned = true;
*/
  t.Stop();
  std::cout << "preprocessing time = " << t.Millisecs() << " ms.\n";
}
void LearningGraph::alloc_on_device() {}
void LearningGraph::alloc_on_device(index_t n) {}
void LearningGraph::copy_to_gpu() {}
