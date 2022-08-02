// Copyright 2022 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "converter.h"

Converter::Converter(std::string file_type, std::string file_name, bool is_bipartite) {
  if (file_type == "mtx")
    read_mtx(file_name, is_bipartite);
  else 
    read_lg(file_name);

  EdgeList el(edge_set.begin(), edge_set.end());
  std::cout << "edgelist size: " << el.size() << "\n";
  ne = el.size(); // remove redundant edges
  printf("|V| %d |E| %ld\n", nv, ne);
  //for (eidType i = 0; i < 10; i ++)
  //  std::cout << "edges[" << i << "]=" << el[i].to_string() << "\n";

  // build CSR
  auto degrees = CountDegrees(nv, el);
  auto max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  std::cout << "maximum degree: " << max_degree << "\n";
  std::vector<vidType> offsets = utils::PrefixSum(degrees);
  rowptr.resize(nv+1);
  for (vidType i = 0; i < nv+1; i ++) rowptr[i] = offsets[i];
  //for (eidType i = 0; i < 10; i ++)
  //  std::cout << "rowptr[" << i << "]=" << rowptr[i] << "\n";
  colidx.resize(ne);
  weights.resize(ne);
  MakeCSR(el, offsets, colidx, weights, has_edge_weights);
  if (has_edge_weights) {
    //for (eidType i = 0; i < 10; i ++)
    //  std::cout << "weights[" << i << "]=" << weights[i] << "\n";
    auto max_elabel = *(std::max_element(weights.begin(), weights.end()));
    auto min_elabel = *(std::min_element(weights.begin(), weights.end()));
    std::cout << "maximum edge lable: " << max_elabel << "\n";
    std::cout << "minimum edge lable: " << min_elabel << "\n";
    elabels.resize(ne);
    for (eidType i = 0; i < ne; i ++)
      elabels[i] = NewEdgeValueT(weights[i]);
    std::set<NewEdgeValueT> labels;
    for (eidType e = 0; e < ne; e++)
      labels.insert(elabels[e]);
    std::cout << "# distinct edge labels: " << labels.size() << "\n";
  }
  /*
  vidType v = 3629;
  auto start = rowptr[v];
  auto end = rowptr[v+1];
  for (eidType i = start; i < end; i ++) {
    std::cout << "edge[" << v << ", " << colidx[i] << "], ";
    std::cout << "elabels=" << unsigned(elabels[i]) << "\n";
  }
  */
}

void Converter::generate_binary_graph(std::string outfilename, bool v, bool e, bool vl, bool el) {
  if (v) {
    std::ofstream outfile((outfilename+".vertex.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(rowptr.data()), (nv+1)*sizeof(eidType));
    outfile.close();
  }

  if (e) {
    std::ofstream outfile1((outfilename+".edge.bin").c_str(), std::ios::binary);
    if (!outfile1) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile1.write(reinterpret_cast<const char*>(colidx.data()), ne*sizeof(vidType));
    outfile1.close();
  }

  if (vl) {
    std::ofstream outfile((outfilename+".vlabel.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(vlabels.data()), nv*sizeof(vlabel_t));
    outfile.close();
  }

  if (el && has_edge_weights) {
    std::ofstream outfile((outfilename+".elabel.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(elabels.data()), ne*sizeof(NewEdgeValueT));
    outfile.close();
  }
}

std::vector<vidType> Converter::CountDegrees(vidType n, EdgeList el, bool symmetrize, bool transpose) {
  std::vector<vidType> degrees(n, 0);
  for (auto e : el) {
    if (symmetrize || (!symmetrize && !transpose))
      degrees[e.src] ++;
    if (symmetrize || (!symmetrize && transpose))
      degrees[e.dst] ++;
  }
  return degrees;
}

void Converter::MakeCSR(EdgeList el, std::vector<vidType> offsets, 
                        VertexList &colidx, std::vector<OldEdgeValueT> &weight, 
                        bool has_edge_weights, bool symmetrize, bool transpose) {
  for (auto e : el) {
    //if (e.label < 1.0)
    //  std::cout << "elabel" << "=" << e.label << "\n";
    if (symmetrize || (!symmetrize && !transpose)) {
      if (has_edge_weights) weight[offsets[e.src]] = e.label;
      colidx[offsets[e.src]++] = e.dst;
    }
    if (symmetrize || (!symmetrize && transpose)) {
      if (has_edge_weights) weight[offsets[e.dst]] = e.label;
      colidx[offsets[e.dst]++] = e.src;
    }
  }
}

void Converter::read_lg(std::string infile_name) {
  std::cout << "Reading TXT/LG file " << infile_name << "\n";
  std::ifstream infile;
  infile.open(infile_name);
  char line[1024];
  std::vector<std::string> result;
  while (true) {
    unsigned pos = infile.tellg();
    if (!infile.getline(line, 1024)) break;
    result.clear();
    utils::split(line, result);
    if (result.empty()) {
    } else if (result[0] == "t") {
      //nv = atoi(result[1].c_str());
      //ne = atoi(result[2].c_str());
      //printf("|V| %d |E| %ld\n", nv, ne);
      //vlabels.resize(nv);
      if (!vlabels.empty()) {   // use as delimiter
        infile.seekg(pos, std::ios_base::beg);
        break;
      } else { }
    } else if (result[0] == "v" && result.size() >= 3) {
      unsigned id = atoi(result[1].c_str());
      vlabels.resize(id + 1);
      //vlabels[id] = atoi(result[2].c_str());
    } else if (result[0] == "e" && result.size() >= 4) {
      vidType from   = atoi(result[1].c_str());
      vidType to     = atoi(result[2].c_str());
      OldEdgeValueT elabel = atoi(result[3].c_str());
      Edge<OldEdgeValueT> edge(from, to, elabel);
      if (edge_set.find(edge) == edge_set.end()) {
        edge_set.insert(edge);
        edge_set.insert(Edge<OldEdgeValueT>(to, from, elabel));
      }
    }
  }
  nv = vlabels.size();
  ne = edge_set.size();
}

void Converter::read_mtx(std::string infile_name, bool is_bipartite) {
  std::cout << "Reading MTX file " << infile_name << "\n";
  std::ifstream infile;
  infile.open(infile_name, std::ios::in);
  bool read_weights;
  std::string start, object, format, field, symmetry, line;
  infile >> start >> object >> format >> field >> symmetry >> std::ws;
  if (start != "%%MatrixMarket") {
    std::cout << ".mtx file did not start with %%MatrixMarket" << std::endl;
    std::exit(-21);
  }
  if ((object != "matrix") || (format != "coordinate")) {
    std::cout << "only allow matrix coordinate format for .mtx" << std::endl;
    std::exit(-22);
  }
  if (field == "complex") {
    std::cout << "do not support complex weights for .mtx" << std::endl;
    std::exit(-23);
  }
  if (field == "pattern") {
    read_weights = false;
  } else if ((field == "real") || (field == "double") || (field == "integer")) {
    read_weights = true;
    std::cout << "edge weights type: " << field << "\n";
  } else {
    std::cout << "unrecognized field type for .mtx" << std::endl;
    std::exit(-24);
  }
  bool undirected;
  if (symmetry == "symmetric") {
    undirected = true;
    std::cout << "This is a symmetric/undirected graph" << std::endl;
  } else if ((symmetry == "general") || (symmetry == "skew-symmetric")) {
    std::cout << "This is an unsymmetric/directed graph" << std::endl;
    undirected = false;
  } else {
    std::cout << "unsupported symmetry type for .mtx" << std::endl;
    std::exit(-25);
  }
  while (true) {
    char c = infile.peek();
    if (c == '%') { infile.ignore(200, '\n');
    } else { break; }
  }
  int64_t m, n, nonzeros;
  infile >> m >> n >> nonzeros >> std::ws;
  std::cout << "m=" << m << " n=" << n << " nnz=" << nonzeros << std::endl;
  if (is_bipartite) {
    nv = m + n;
    std::cout << "Bipartite graph\n";
  } else {
    nv = m;
    //std::cout << "non-Bipartite graph\n";
    if (m != n) {
      std::cout << "matrix must be square for .mtx unless it is a bipartite graph" << std::endl;
      std::exit(-26);
    }
  }
  if (is_bipartite) undirected = true;
  eidType idx = 0;
  int num_redundant_edges = 0;
  edge_set.clear();
  while (std::getline(infile, line)) {
    std::istringstream edge_stream(line);
    vidType u, v;
    edge_stream >> u;
    edge_stream >> v;
    auto src = u-1;
    auto dst = v-1;
    if (is_bipartite) dst += m;
    if (src == dst) continue; // remove selfloops
    OldEdgeValueT elabel = 0;
    if (read_weights) edge_stream >> elabel;
    //std::cout << "src " << src << " dst " << dst << " elabel " << elabel << "\n";
    Edge<OldEdgeValueT> edge(src, dst, elabel);
    if (edge_set.find(edge) == edge_set.end()) { // remove redundant edges
      edge_set.insert(edge);
      if (undirected) {
        Edge<OldEdgeValueT> reverse_edge(dst, src, elabel);
        edge_set.insert(reverse_edge);
      }
    } else {
      num_redundant_edges ++;
    }
    idx ++;
  }
/*
  char line[1024];
  std::vector<std::string> result;
  while (infile.getline(line, 1024)) {
    result.clear();
    utils::split(line, result);
    if (result.empty() || result[0] == "%") {
      // ignore
    } else {
      vidType src = atoi(result[0].c_str()) - 1;
      vidType dst = atoi(result[1].c_str()) - 1;
      if (bipartite) dst += m;
      OldEdgeValueT elabel = 0;
      if (result.size() > 2) elabel = atof(result[2].c_str());
      //if (elabel < 1)
      //  std::cout << "src " << from << " dst " << to << " elabel " << elabel << "\n";
      Edge<OldEdgeValueT> edge(src, dst, elabel);
      Edge<OldEdgeValueT> reverse_edge(dst, src, elabel);
      if (edge_set.find(edge) == edge_set.end()) {
        edge_set.insert(edge);
        if (undirected) {
          edge_set.insert(reverse);
        }
      } else {
        //std::cout << "redundant edge: ";
        //std::cout << "src " << from << " dst " << to << " elabel " << elabel << "\n";
        //exit(0);
      }
      idx ++;
    }
  }
  */
  std::cout << "Complete reading " << idx << " lines/edges\n";
  std::cout << "Found " << idx - num_redundant_edges << " distinct edges and " 
            << num_redundant_edges << " redundant edges\n";
}

