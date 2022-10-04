#pragma once

#include <queue>
#include <unordered_set>
#include <vector>

#include "pam/pam.h"
#include "graph.h"
#include "parlay/sequence.h"

#define UINT_E_MAX UINT_MAX

// Alias template for parlay::sequence
template <typename T>
using sequence = parlay::sequence<T>;

namespace approx_average_linkage {

//template <class Weights, class IW, template <class W> class w_vertex>
template <class Weights>
struct clustered_graph {
  //using orig_vertex = w_vertex<IW>;
  using W = typename Weights::weight_type;
  using internal_edge = std::pair<vidType, std::pair<vidType, W>>;
  using edge = std::pair<vidType, W>;

  struct neighbor_entry {
    using key_t = vidType;                // neighbor_id
    using val_t = std::pair<vidType, W>;  // (id * weight)
    using aug_t = W;                    // aggregated weight
    static inline bool comp(key_t a, key_t b) { return a < b; }
    static aug_t get_empty() { return Weights::id(); }
    static aug_t from_entry(key_t k, val_t v) {
      return v.second;
    }  // (get weight)
    // used to select min/max edges based on similarity/dissimilarity
    // clustering.
    static aug_t combine(aug_t a, aug_t b) {
      return Weights::augmented_combine(a, b);
    }
  };

  using neighbor_map = aug_map<neighbor_entry>;

  struct clustered_vertex {
    clustered_vertex() {}

    clustered_vertex(vidType vtx_id, Graph& G,
                     const Weights& weights) {
      auto cluster_size = G.out_degree(vtx_id);
      staleness = cluster_size;
      num_in_cluster = 1;  // initially just this vertex
      active = true;
      current_id = vtx_id;

      auto edges = sequence<internal_edge>::uninitialized(cluster_size);

      size_t i = 0;
      auto map_f = [&](const vidType& u, const vidType& v, const elabel_t& wgh) {
        W true_weight = Weights::get_weight(u, v, wgh);
        edges[i++] = std::make_pair(v, std::make_pair(v, true_weight));
      };
      //vertex.out_neighbors().map(map_f, /* parallel = */ false);
      vidType j = 0;
      for (auto u: G.N(vtx_id)) {
        map_f(vtx_id, u, G.get_elabel(vtx_id, j));
        j++;
      }

      neighbors = neighbor_map(edges);
    }

    std::optional<edge> highest_priority_edge() {
      if (neighbor_size() == 0) return {};
      W m = neighbors.aug_val();
      internal_edge entry;
      entry = *neighbors.aug_eq(m);
      assert(entry.second.second == m);
      return entry.second;
    }

    vidType neighbor_size() { return neighbors.size(); }

    vidType size() { return num_in_cluster; }

    bool is_active() { return active; }

    vidType get_current_id() { return current_id; }

    void set_current_id(vidType id) { current_id = id; }

    bool is_stale(double epsilon) {
      return ((staleness * (1 + epsilon)) < size());
    }

    // Tracks the last cluster update size.
    vidType staleness;
    // The "current" id of this cluster, updated upon a merge that keeps this
    // cluster active.
    vidType current_id;
    // Number of vertices contained in this cluster.
    vidType num_in_cluster;
    // Active == false iff this cluster has not yet been clustered
    bool active;
    // An augmented map storing our neighbors + weights.
    neighbor_map neighbors;
  };

  Graph& G;
  Weights& weights;
  double epsilon;
  vidType n;
  vidType last_cluster_id;
  vidType num_merges_performed;

  parlay::sequence<clustered_vertex> clusters;
  parlay::sequence<std::pair<vidType, W>> dendrogram;

  // Returns whether this cluster is still active, or whether it has been merged
  // into a _larger_ cluster.
  bool is_active(vidType id) { return clusters[id].is_active(); }

  vidType new_cluster_id() {
    vidType ret = last_cluster_id;
    last_cluster_id++;
    return ret;
  }

  vidType unite(vidType a, vidType b, W wgh) {
    assert(is_active(a));
    assert(is_active(b));
    // Identify smaller/larger clusters (will merge smaller -> larger).
    vidType d_a = clusters[a].neighbor_size();
    vidType d_b = clusters[b].neighbor_size();
    vidType smaller, larger;
    if (d_a < d_b) {
      smaller = a;
      larger = b;
    } else {
      larger = a;
      smaller = b;
    }

    // Deactivate smaller.
    clusters[smaller].active = false;

    // Merge smaller and larger's neighbors.
    auto smaller_ngh = std::move(clusters[smaller].neighbors);
    auto larger_ngh = std::move(clusters[larger].neighbors);

    // Some sanity asserts, we are merging an edge incident to both after all.
    assert(smaller_ngh.size() > 0);
    assert(larger_ngh.size() > 0);

    // Remove larger's id from smaller, and vice versa.
    assert(smaller_ngh.contains(larger));
    assert(larger_ngh.contains(smaller));
    auto small_pre_merge = neighbor_map::remove(std::move(smaller_ngh), larger);
    auto large_pre_merge = neighbor_map::remove(std::move(larger_ngh), smaller);

    auto smaller_keys = neighbor_map::keys(small_pre_merge);

    // First merge to calculate the new size of this cluster.
    auto first_merge =
        neighbor_map::map_union(small_pre_merge, large_pre_merge);
    //debug(vidType merged_size = first_merge.size(););
    first_merge.~neighbor_map();

    size_t new_cluster_size =
        clusters[larger].num_in_cluster + clusters[smaller].num_in_cluster;
    auto linkage = Weights::GetLinkage(clusters, new_cluster_size);
    std::cout << "Performed first merge. New cluster size = "
              << new_cluster_size << std::endl;

    auto merged = neighbor_map::map_union(std::move(small_pre_merge),
                                          std::move(large_pre_merge), linkage);
    //assert(merged.size() == merged_size);

    clusters[larger].neighbors = std::move(merged);

    // Save that clusters a and b are merged.
    vidType current_a = clusters[a].get_current_id();
    vidType current_b = clusters[b].get_current_id();
    vidType new_id = new_cluster_id();  // increments next_id
    num_merges_performed++;

    dendrogram[current_a] = {new_id, wgh};
    dendrogram[current_b] = {new_id, wgh};

    // Update the current id of the remaining cluster.
    clusters[larger].current_id = new_id;

    // Update the size of the remaining cluster.
    clusters[larger].num_in_cluster = new_cluster_size;
    std::cout << "Num in cluster = " << clusters[larger].num_in_cluster
              << std::endl;

    // Map over _all_ of smaller's edges, and update its neighbors to point to
    // larger. If the neighbor, w, also has an edge to larger (a
    // smaller-larger-w triangle), then update the weight of this edge.
    for (size_t i = 0; i < smaller_keys.size(); i++) {
      vidType w = smaller_keys[i];
      assert(clusters[w].neighbors.contains(smaller));  // Sanity.

      auto w_zero = std::move(clusters[w].neighbors);
      auto found_value = *(w_zero.find(smaller));  // value
      auto w_one = neighbor_map::remove(std::move(w_zero), smaller);

      // Insert larger, merging using Weights::linkage if it already exists in
      // the tree.
      found_value.first = larger;
      auto new_value =
          Weights::UpdateWeight(clusters, found_value, new_cluster_size);
      auto larger_ent = std::make_pair(larger, new_value);

      w_one.insert(larger_ent, linkage);

      // Move the neighbors back.
      clusters[w].neighbors = std::move(w_one);
    }

    // Staleness check.
    if (clusters[larger].is_stale(epsilon)) {
      std::cout << "LARGER = " << larger << " is STALE" << std::endl;
      // Update our own edges.
      auto edges = std::move(clusters[larger].neighbors);
      auto map_f = [&](const auto& entry) {
        return Weights::UpdateWeight(clusters, entry.second, new_cluster_size);
      };
      auto updated_edges = neighbor_map::map(edges, map_f);
      clusters[larger].neighbors = std::move(updated_edges);

      // Map over the edges, and update on our neighbors endpoints.
      auto update_ngh_f = [&](const auto& entry) {
        vidType ngh_id = entry.first;
        auto val = entry.second;
        //debug(vidType val_id = val.first; assert(ngh_id == val_id););

        val.first = larger;  // place our id
        auto updated_val = Weights::UpdateWeight(
            clusters, val, new_cluster_size);  // update weight
        auto new_entry = std::make_pair(larger, updated_val);

        // Now update our neighbor.
        assert(clusters[ngh_id].neighbors.contains(larger));
        clusters[ngh_id].neighbors.insert(new_entry);
      };
      neighbor_map::map_void(clusters[larger].neighbors, update_ngh_f);

      // Update staleness.
      clusters[larger].staleness = clusters[larger].size();
      std::cout << "Finished update." << std::endl;
    }

    return larger;
  }

  clustered_graph(Graph& G, Weights& weights, double epsilon)
      : G(G), weights(weights), epsilon(epsilon) {
    n = G.V();
    last_cluster_id = n;
    num_merges_performed = 0;
    clusters = parlay::sequence<clustered_vertex>(n);
    dendrogram = parlay::sequence<std::pair<vidType, W>>(
        2 * n - 2, std::make_pair(UINT_E_MAX, W()));

    parlay::parallel_for(0, n, [&](size_t i) {
      clusters[i] = clustered_vertex(i, G, weights);
    });
    std::cout << "Built all vertices" << std::endl;
  }

  // extract dendrogram
  sequence<std::pair<vidType, W>> get_dendrogram() {
    std::cout << "num_merges_performed = " << num_merges_performed << std::endl;
    std::cout << "n = " << n << std::endl;

    if (num_merges_performed < n - 1) {
      size_t last_clust = last_cluster_id;
      auto ids = parlay::delayed_seq<vidType>(last_clust + 1, [&](size_t i) {
        if (dendrogram[i].first == UINT_E_MAX) return (vidType)i;
        return UINT_E_MAX;
      });
      auto bad =
          parlay::filter(ids, [&](const vidType& e) { return e != UINT_E_MAX; });

      std::cout << "num bad = " << bad.size() << std::endl;

      std::queue<vidType> bad_queue;
      for (size_t i = 0; i < bad.size(); i++) {
        bad_queue.push(bad[i]);
      }

      while (bad_queue.size() > 1) {
        vidType fst = bad_queue.front();
        bad_queue.pop();
        vidType snd = bad_queue.front();
        bad_queue.pop();

        vidType new_id = new_cluster_id();  // increments next_id
        dendrogram[fst] = {new_id, Weights::id()};
        dendrogram[snd] = {new_id, Weights::id()};

        std::cout << "Merged components for: " << fst << " " << snd
                  << std::endl;

        bad_queue.push(new_id);
      }
    }

    return std::move(dendrogram);
  }
};

}  // namespace clustering
