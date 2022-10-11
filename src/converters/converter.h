#include "utils.h"
#include "graph.h"
// convert txt (edgelist) to binary (CSR)
typedef float OldEdgeValueT;
typedef float NewEdgeValueT;
typedef std::pair<vidType,OldEdgeValueT> Neighbor;
//typedef elabel_t NewEdgeValueT;

struct Edge {
  vidType src;
  vidType dst;

  Edge(vidType s, vidType d) {
    src = s; dst = d;
  }
  void operator = (const Edge &E) { 
    src = E.src;
    dst = E.dst;
  }
  bool operator == (const Edge &E) {
    if (src == E.src && dst == E.dst) return true;
    return false;
  }
  std::string to_string() {
    return "<" + std::to_string(src) + "," + std::to_string(src) + ">";
  }
};

template <typename T>
inline bool operator < (const Edge &E1, const Edge &E2) {
  if (E1.src < E2.src || (E1.src == E2.src && E1.dst < E2.dst)) return true;
  return false;
}

template <typename T>
struct WeightedEdge {
  vidType src;
  vidType dst;
  T label;

  WeightedEdge(vidType s, vidType d, T l) {
    src = s; dst = d; label = l;
  }
  void operator = (const WeightedEdge &E) { 
    src = E.src;
    dst = E.dst;
    label = E.label;
  }
  bool operator == (const WeightedEdge &E) {
    if (src == E.src && dst == E.dst && label == E.label) return true;
    return false;
  }
  std::string to_string() {
    return "<" + std::to_string(src) + "," + std::to_string(src) + "," + std::to_string(label) + ">";
  }
};

template <typename T>
inline bool operator < (const WeightedEdge<T> &E1, const WeightedEdge<T> &E2) {
  if (E1.src < E2.src || (E1.src == E2.src && E1.dst < E2.dst)) return true;
  return false;
}

typedef std::vector<Edge> EdgeList;
typedef std::vector<WeightedEdge<OldEdgeValueT>> WeightedEdgeList;
typedef std::set<Edge> EdgeSet;
typedef std::set<WeightedEdge<OldEdgeValueT>> WeightedEdgeSet;
typedef std::set<vidType> NeighborSet;
typedef std::set<Neighbor> WeightedNeighborSet;

class Converter {
public:
  Converter() : nv(0), ne(0), has_edge_weights(false), g(NULL) {}
  Converter(std::string file_type, std::string file_name, bool is_bipartite);
  void read_edgelist(std::string infile_name);
  void read_sadj(std::string infile_name);
  void read_lg(std::string filename);
  void read_mtx(std::string filename, bool is_bipartite);
  void generate_binary_graph(std::string outfilename, bool v = true, bool e = true, bool vl = true, bool el = true);
  void readGraphFromGRFile(std::string filename, bool need_sort = false);
  void read_labels(std::string filename, size_t num_classes, bool is_single_class);
  size_t read_masks(std::string mask_type, std::string filename, size_t begin_, size_t end_, mask_t* masks);
  void splitGRFile(std::string filename, std::string outfilename);

private:
  uint64_t nv;
  uint64_t ne;
  bool has_edge_weights;
  std::vector<vidType> degrees;
  std::vector<vlabel_t> vlabels;
  std::vector<OldEdgeValueT> weights;
  std::vector<NewEdgeValueT> elabels;
  EdgeSet edge_set;
  WeightedEdgeSet weighted_edge_set;
  std::vector<NeighborSet> adj_lists;
  std::vector<WeightedNeighborSet> weighted_adj_lists;
  Graph *g;

  void edgelist2CSR();
  void weighted_edgelist2CSR();
  void adjlist2CSR();
  void weighted_adjlist2CSR();
  void CountDegrees(WeightedEdgeList el, bool symmetrize = false, bool transpose = false);
  void split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ");
};

