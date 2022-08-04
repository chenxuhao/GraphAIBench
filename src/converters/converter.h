#include "utils.h"
// convert txt (edgelist) to binary (CSR)
typedef float OldEdgeValueT;
typedef float NewEdgeValueT;
//typedef elabel_t NewEdgeValueT;

template <typename T>
struct Edge {
  Edge(vidType s, vidType d, T l) {
    src = s; dst = d; label = l; }
  vidType src;
  vidType dst;
  T label;
  void operator = (const Edge &E) { 
    src = E.src;
    dst = E.dst;
    label = E.label;
  }
  bool operator == (const Edge &E) {
    if (src == E.src && dst == E.dst && label == E.label) return true;
    return false;
  }
  std::string to_string() {
    return "<" + std::to_string(src) + "," + std::to_string(src) + "," + std::to_string(label) + ">";
  }
};

template <typename T>
inline bool operator < (const Edge<T> &E1, const Edge<T> &E2) {
  if (E1.src < E2.src || (E1.src == E2.src && E1.dst < E2.dst)) return true;
  return false;
}

typedef std::vector<Edge<OldEdgeValueT>> EdgeList;
typedef std::set<Edge<OldEdgeValueT>> EdgeSet;

class Converter {
public:
  Converter(std::string file_type, std::string file_name, bool is_bipartite);
  void read_lg(std::string filename);
  void read_mtx(std::string filename, bool is_bipartite);
  void generate_binary_graph(std::string outfilename, bool v = true, bool e = true, bool vl = true, bool el = true);

private:
  vidType nv;
  eidType ne;
  bool has_edge_weights;
  std::vector<eidType> rowptr;
  std::vector<vidType> colidx;
  std::vector<vlabel_t> vlabels;
  std::vector<OldEdgeValueT> weights;
  std::vector<NewEdgeValueT> elabels;
  EdgeSet edge_set;

  std::vector<vidType> CountDegrees(vidType n, EdgeList el, bool symmetrize = false, bool transpose = false);
  void MakeCSR(EdgeList el, std::vector<vidType> offsets, 
               VertexList &colidx, std::vector<OldEdgeValueT> &weight, 
               bool has_edge_weights = false,
               bool symmetrize = false, 
               bool transpose = false);
};

