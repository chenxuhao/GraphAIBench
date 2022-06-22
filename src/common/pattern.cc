#include "pattern.hh"
#include "scan.h"

void Pattern::set_name() {
  auto n = n_vertices;
  auto m = n_edges / 2;
  name_ = "";
  if (num_vertex_classes > 0) 
    name_ += std::to_string(num_vertex_classes) + "labeled-";
  if (n == 3) {
    if (m == 2) name_ += "wedge";
    else name_ += "triangle";
  } else if (n == 4) {
    if (m == 3) {
      if (max_degree == 3) name_ += "3-star";
      else name_ = "4-path";
    } else if (m == 4) {
      if (max_degree == 3) name_ += "tailed_triangle";
      else name_ += "square";
    } else if (m == 5) {
      name_ += "diamond";
    } else {
      assert(m==6);
      name_ += "4-clique";
    }
  } else {
    name_ += "unknown";
  }
}

std::vector<vidType> Pattern::v_list() const {
  std::vector<vidType> vs;
  for (auto pair : adj_list) vs.push_back(pair.first);
  std::sort(vs.begin(), vs.end());
  return vs;
}

std::string Pattern::to_string(const std::vector<vlabel_t> &given_labels) const {
  if (num_vertex_classes > 0) {
    assert(given_labels.size() >= size_t(n_vertices));
    std::string res("");
    for (auto pair : adj_list) {
      auto u = pair.first;
      auto l1 = given_labels[u] == static_cast<vlabel_t>(-1)
        ? "*" : std::to_string(given_labels[u]);
      for (auto v : pair.second) {
        if (u > v) continue;
        auto l2 = given_labels[v] == static_cast<vlabel_t>(-1)
          ? "*" : std::to_string(given_labels[v]);
        res += "[";
        res += std::to_string(u) + "," + l1;
        res += "-";
        res += std::to_string(v) + "," + l2;
        res += "]";
      }
    }
    return res;
  } else {
    return to_string();
  }
}

std::string Pattern::to_string() const {
  std::string res("");
  if (num_vertex_classes > 0) {
    for (vidType v = 0; v < n_vertices; ++v) {
      res += vlabels[v];
    }
  } else {
    for (auto pair : adj_list) {
      auto u = pair.first;
      for (auto v : pair.second) {
        if (u > v) continue;
        res += "[";
        res += std::to_string(u);
        res += "-";
        res += std::to_string(v);
        res += "]";
      }
    }
  }
  return res;
}

void Pattern::read_adj_file(std::string inputfile) {
  //std::cout << "Reading pattern graph from file: " << inputfile << "\n";
  std::ifstream query_graph(inputfile.c_str());
  query_graph >> n_vertices >> n_edges >> max_degree >> num_vertex_classes >> num_edge_classes;
  //std::cout << "|V|: " << n_vertices << " |E|: " << n_edges << " max_degree: " << max_degree
  //          << " vertex-|\u03A3|: " << num_vertex_classes
  //          << " edge-|\u03A3|: " << num_edge_classes << "\n";
  if (num_vertex_classes > 0) vlabels = new vlabel_t[n_vertices]; 
  if (num_edge_classes > 0) elabels = new elabel_t[n_edges];
  // TODO: edge labels not supported yet
  std::string line;
  while (std::getline(query_graph, line)) {
    std::istringstream iss(line);
    std::vector<int> vs(std::istream_iterator<int>{iss}, std::istream_iterator<int>());
    if (vs.size() == 0) continue;
    int v = vs[0];
    int vl = vs[1];
    //std::cout << "v " << v << " label " << vl << "\n";
    if (num_vertex_classes > 0) vlabels[v] = vl;
    for (size_t i = 2; i < vs.size(); i++)
      adj_list[v].push_back(vs[i]);
  }
  assert(size_t(n_vertices) == adj_list.size());
  int ne = 0;
  for (auto pair : adj_list) ne += pair.second.size();
  assert(n_edges == ne);
  int md = 0;
  for (vidType v = 0; v < n_vertices; ++v) {
    auto deg = get_degree(v);
    if (deg > md)
      md = deg;
  }
  assert(md == max_degree);
  generateCSR();
  computeLabelsFrequency();
}

void Pattern::generateCSR() {
  vertices = custom_alloc_global<eidType>(n_vertices+1);
  edges = custom_alloc_global<vidType>(n_edges);
  std::vector<vidType> degrees(n_vertices, 0);
  for (vidType v = 0; v < n_vertices; v++)
    degrees[v] = get_degree(v);
  parallel_prefix_sum<vidType,eidType>(degrees, vertices);
  for (vidType v = 0; v < n_vertices; ++v) {
    auto begin = vertices[v];
    eidType offset = 0;
    for (auto u : N(v)) {
      edges[begin+offset] = u;
      offset ++;
    }
  }
}

void Pattern::analyze() {
  auto n = n_vertices;
  auto m = n_edges / 2;
  assert(n > 2);
  num_operators.resize(n-2);
  for (int i = 0; i < n-2; i++)
    num_operators[i] = i+1;
  set_operators.resize(n-2);
  set_operands.resize(n-2);
  for (int i = 0; i < n-2; i++) {
    set_operators[i].resize(num_operators[i]);
    set_operands[i].resize(num_operators[i]+1);
  }
  set_operands[0][0] = 0;
  set_operands[0][1] = 1;
  set_operators[0][0] = SET_INTERSECTION;
  if (n > 3) {
    set_operands[1][0] = 0;
    set_operands[1][1] = 1;
    set_operands[1][2] = 2;
    set_operators[1][0] = SET_INTERSECTION;
    set_operators[1][1] = SET_INTERSECTION;
  }
  if (n == 3) {
    if (m == 2)
      set_operators[0][0] = SET_DIFFERENCE;
  } else if (n == 4) {
    if (m == 3) { // 3-star or 4-path
      set_operators[0][0] = SET_DIFFERENCE;
      set_operators[1][0] = SET_DIFFERENCE;
      set_operators[1][1] = SET_DIFFERENCE;
      if (max_degree == 3) { // 3-star
      } else { // 4-path
        set_operands[1][0] = 2;
        set_operands[1][1] = 0;
        set_operands[1][2] = 1;
      }
    } else if (m == 4) {
      if (max_degree == 3) { // tailed_triangle
        set_operators[1][0] = SET_DIFFERENCE;
        set_operators[1][1] = SET_DIFFERENCE;
      } else { // square
        set_operators[0][0] = SET_DIFFERENCE;
        set_operators[1][1] = SET_DIFFERENCE;
        set_operands[1][0] = 1;
        set_operands[1][1] = 2;
        set_operands[1][2] = 0;
      }
    } else if (m == 5) {
        set_operators[1][1] = SET_DIFFERENCE;
      // diamond
    } else {
      // 4-clique
    }
  }
  std::cout << "the level-2 set operator: " << set_operators[0][0] << "\n";
  if (n>3) {
    std::cout << "the first level-3 set operator: " << set_operators[1][0] << "\n";
    std::cout << "the second level-3 set operator: " << set_operators[1][1] << "\n";
  }
}

bool Pattern::is_connected(vidType u, vidType v) const {
  if (get_degree(u) < get_degree(v)) std::swap(u, v);
  int begin = 0;
  int end = get_degree(v)-1;
  while (begin <= end) {
    int mid = begin + ((end - begin) >> 1);
    auto w = get_neighbor(v, mid);
    if (w == u) return true;
    else if (w > u) end = mid - 1;
    else begin = mid + 1;
  }
  return false;
}

void Pattern::add_edge(vidType u, vidType v, elabel_t el) {
  adj_list[u].push_back(v);
  adj_list[v].push_back(u);
  if (has_elabel()) { // TODO
    //elabels.push_back(static_cast<elabel_t>(el));
  }
}

