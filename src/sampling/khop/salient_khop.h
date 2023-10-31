// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "utils.h"
#include "graph.h"
#include <random>
std::mt19937 gen(time(nullptr));


/**
 * Creates the initial seeds.
 *
 * @param seeds_size size of initial sample
 * @param graph_size size of original graph
 * @return vector containing node ids
*/
inline std::vector<uint32_t> get_initial_sample(uint64_t seeds_size, uint64_t graph_size) {
  std::set<uint32_t> node_ids;
  std::vector<uint32_t> n_ids;
  for (size_t i = 0; i < seeds_size; i++) {
    auto sample_id = gen() % graph_size;
    node_ids.insert(sample_id);
  }
  n_ids.insert(n_ids.end(), node_ids.begin(), node_ids.end());
  return n_ids;
}

/**
 * As subgraph grows, every new added node is relabeled to be
 * the order it was added in. This function creates the new
 * labels for the initial seeds in the subgraph, and we add to
 * the returned map as the subgraph grows.
 *
 * @param n_ids initial seed node ids
 * @return mapping of node ids in original graph to new subgraph
*/
inline std::unordered_map<uint32_t, uint32_t> get_initial_sample_adj_hash_map(const std::vector<uint32_t>& n_ids) {
  // for initial seeds, map node id to their position in n_ids
  std::unordered_map<uint32_t, uint32_t> n_id_map;
  for (size_t i = 0; i < n_ids.size(); ++i) {
    n_id_map.insert({n_ids[i], i});
  }
  return n_id_map;
}

/**
 * Does one round of sampling from given seeds
 *
 * @param rowptr row pointers of csr format for entire graph
 * @param col column indices of csr format for entire graph
 * @param n_ids seed ids (same as rows)
 * @param all_ids all already added ids to subgraph
 * @param n_id_map maps new ids to old
 * @param num_neighbors number of new samples to take per seed
 * @param replace true if sample with replacement, false o/w
 * @return (rowptrs of new samples, col indices of new samples, n_ids)
*/
inline std::tuple<std::vector<vidType, std::allocator<vidType>>, std::vector<vidType, std::allocator<vidType>>, std::vector<vidType, std::allocator<vidType>>> sample_adj(vector<eidType> rowptr, vector<vidType> col, std::vector<uint32_t> n_ids, std::vector<uint32_t>& all_ids, std::unordered_map<uint32_t, uint32_t>& n_id_map, uint32_t num_neighbors, bool replace) {

  // number of seeds to add
  const uint32_t idx_size = n_ids.size();

  // store new sample row pointers
  vector<uint32_t> out_rowptr;
  out_rowptr.push_back(0);

  // adjacency matrix for new samples
  std::vector<std::vector<uint32_t>> cols(idx_size);

  const auto expand_neighborhood = [&](auto add_neighbors) -> void {
    for (size_t i = 0; i < idx_size; ++i) {
      // pick seed to sample neighbors from
      const uint32_t n = n_ids[i];
      const eidType row_start = rowptr[n];
      const eidType row_end = rowptr[n + 1];
      const uint64_t neighbor_count = row_end - row_start;

      const auto add_neighbor = [&](const uint32_t p) -> void {
        const eidType e = row_start + p;
        const vidType c = col[e];

        // builds mapping
        auto ins = n_id_map.insert({c, all_ids.size()});
        // add to all_ids to track size for mapping
        if (ins.second) {
          all_ids.push_back(c);
        }
        // store new node adjacencies in terms of old ids
        cols[i].push_back(ins.first->first);
      };

      add_neighbors(neighbor_count, add_neighbor);
      out_rowptr.push_back(out_rowptr[i] + cols[i].size());
    }
  };

  if (num_neighbors <
      0) {  // No sampling ======================================
      // returns entire input graph
    expand_neighborhood([](const uint32_t neighbor_count, auto add_neighbor) {
      for (uint32_t j = 0; j < neighbor_count; j++) {
        add_neighbor(j);
      }
    });
  } else if (replace) {  // Sample with replacement
                         // =============================
    expand_neighborhood(
        [num_neighbors](const uint32_t neighbor_count, auto add_neighbor) {
          if (neighbor_count <= 0) return;
          for (uint32_t j = 0; j < num_neighbors; j++) {
            add_neighbor(gen() % neighbor_count);
          }
        });
  } else {  // Sample without replacement via Robert Floyd algorithm
            // ============
    std::vector<uint32_t> perm;
    perm.reserve(num_neighbors);
    expand_neighborhood([num_neighbors, &perm](const uint32_t neighbor_count,
                                               auto add_neighbor) {
      perm.clear();

      if (neighbor_count <= num_neighbors) {
        for (uint32_t j = 0; j < neighbor_count; j++) {
          add_neighbor(j);
        }
      } else {  // See: https://www.nowherenearithaca.com/2013/05/
                //      robert-floyds-tiny-and-beautiful.html
        for (uint32_t j = neighbor_count - num_neighbors; j < neighbor_count;
             j++) {
          const uint32_t option = gen() % j;
          auto winner = option;
          if (std::find(perm.cbegin(), perm.cend(), option) == perm.cend()) {
            perm.push_back(option);
            winner = option;
          } else {
            perm.push_back(j);
            winner = j;
          }

          add_neighbor(winner);
        }
      }
    });
  }

  // gets column indices in terms of original node ids of subgraph
  vector<uint32_t> out_col;
  // size_t i = 0;
  for (auto& col_vec : cols) {
    std::sort(col_vec.begin(), col_vec.end());
    for (const auto& value : col_vec) {
      // edges that need to be added
      out_col.push_back(value);
      // i += 1;
    }
  }

  return std::make_tuple(std::move(out_rowptr), std::move(out_col), std::move(all_ids));

}