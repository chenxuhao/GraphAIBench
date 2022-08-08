// Copyright 2022 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "converter.h"
#include "scan.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

Converter::Converter(std::string file_type, std::string file_name, bool is_bipartite) : ftype(file_type) {
  if (file_type == "gr" || file_type == "sgr" || file_type == "csgr") {
    readGraphFromGRFile(file_name, true);
  } else {
    if (file_type == "mtx") {
      read_mtx(file_name, is_bipartite);
    } else if (file_type == "edges") {
      // plain edgelist format
      read_edgelist(file_name);
    } else {
      read_lg(file_name);
    }
    edgelist2CSR();
  }
}

void Converter::edgelist2CSR() {
  EdgeList el(edge_set.begin(), edge_set.end());
  std::cout << "edgelist size: " << el.size() << "\n";
  ne = el.size(); // redundant edges removed
  printf("|V| %ld |E| %ld\n", nv, ne);

  // build CSR
  degrees.resize(nv);
  CountDegrees(el);
  g = new Graph(nv, ne);
  auto rowptr = g->rowptr();
  parallel_prefix_sum<vidType,eidType>(degrees, rowptr);
  weights.resize(ne);
  MakeCSR(el, has_edge_weights);
  if (has_edge_weights) {
    auto max_elabel = *(std::max_element(weights.begin(), weights.end()));
    auto min_elabel = *(std::min_element(weights.begin(), weights.end()));
    std::cout << "maximum edge lable: " << max_elabel << "\n";
    std::cout << "minimum edge lable: " << min_elabel << "\n";
    elabels.resize(ne);
    for (eidType i = 0; i < g->E(); i ++)
      elabels[i] = NewEdgeValueT(weights[i]);
    std::set<NewEdgeValueT> labels;
    for (eidType e = 0; e < g->E(); e++)
      labels.insert(elabels[e]);
    std::cout << "# distinct edge labels: " << labels.size() << "\n";
  }
}

void Converter::generate_binary_graph(std::string outfilename, bool v, bool e, bool vl, bool el) {
  if (v) {
    std::ofstream outfile((outfilename+".vertex.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(g->rowptr()), (nv+1)*sizeof(eidType));
    outfile.close();
  }

  if (e) {
    std::ofstream outfile1((outfilename+".edge.bin").c_str(), std::ios::binary);
    if (!outfile1) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile1.write(reinterpret_cast<const char*>(g->colidx()), ne*sizeof(vidType));
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

void Converter::CountDegrees(EdgeList el, bool symmetrize, bool transpose) {
  for (auto e : el) {
    if (symmetrize || (!symmetrize && !transpose))
      degrees[e.src] ++;
    if (symmetrize || (!symmetrize && transpose))
      degrees[e.dst] ++;
  }
  auto max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  std::cout << "maximum degree: " << max_degree << "\n";
}

void Converter::MakeCSR(EdgeList el, bool has_edge_weights, bool symmetrize, bool transpose) {
  auto offsets = g->rowptr();
  auto colidx = g->colidx();
  for (auto e : el) {
    //if (e.label < 1.0)
    //  std::cout << "elabel" << "=" << e.label << "\n";
    if (symmetrize || (!symmetrize && !transpose)) {
      if (has_edge_weights) weights[offsets[e.src]] = e.label;
      colidx[offsets[e.src]++] = e.dst;
    }
    if (symmetrize || (!symmetrize && transpose)) {
      if (has_edge_weights) weights[offsets[e.dst]] = e.label;
      colidx[offsets[e.dst]++] = e.src;
    }
  }
}

void Converter::split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters) {
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
	while (std::string::npos != pos || std::string::npos != lastPos) {
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		lastPos = str.find_first_not_of(delimiters, pos);
		pos = str.find_first_of(delimiters, lastPos);
	}
}

void Converter::read_sadj(std::string infile_name) {
  FILE* fd = fopen(infile_name.c_str(), "r");
  assert (fd != NULL);
  char buf[2048];
  int size = 0, maxsize = 0;
  while (fgets(buf, 2048, fd) != NULL) {
    int len = strlen(buf);
    size += len;
    if (buf[len-1] == '\n') {
      maxsize = std::max(size, maxsize);
      size = 0;
    }
  }
  fclose(fd);

  std::ifstream is;
  is.open(infile_name, std::ios::in);
  char*line = new char[maxsize+1];
  std::vector<std::string> result;
  nv = 0;
  while (is.getline(line, maxsize+1)) {
    nv++;
    result.clear();
    split(line, result);
    int src = atoi(result[0].c_str());
    vlabels.resize(src + 1);
    vlabels[src] = atoi(result[1].c_str());
    std::set<std::pair<vidType, OldEdgeValueT> > neighbors;
    for(size_t i = 2; i < result.size(); i++) {
      vidType dst = atoi(result[i].c_str());
      if (src == dst) continue; // remove self-loop
      neighbors.insert(std::pair<vidType, OldEdgeValueT>(dst, 0)); // remove redundant edge
    }
    for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
      Edge<OldEdgeValueT> edge(src, it->first, it->second);
      if (edge_set.find(edge) == edge_set.end()) {
        edge_set.insert(edge);
      }
    }
  }
  is.close();
}

void Converter::read_edgelist(std::string infile_name) {
  std::cout << "Reading plain edgelist file " << infile_name << "\n";
  std::ifstream infile;
  infile.open(infile_name);
  char line[1024];
  std::vector<std::string> result;
  vidType num = 0;
  while (infile.getline(line, 1024)) {
    result.clear();
    utils::split(line, result);
    assert(result.size() >= 2);
    vidType src = atoi(result[1].c_str());
    vidType dst = atoi(result[2].c_str());
    if (src == dst) continue; // remove self-loop
    if (src+1 > num) num = src+1;
    if (dst+1 > num) num = dst+1;
    OldEdgeValueT elabel = 0;
    if (result.size() > 2) atoi(result[3].c_str());
    Edge<OldEdgeValueT> edge(src, dst, elabel);
    if (edge_set.find(edge) == edge_set.end()) {
      edge_set.insert(edge);
      edge_set.insert(Edge<OldEdgeValueT>(dst, src, elabel));
    }
  }
  nv = num;
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
  std::cout << "Complete reading " << idx << " lines/edges\n";
  std::cout << "Found " << idx - num_redundant_edges << " distinct edges and " 
            << num_redundant_edges << " redundant edges\n";
}

void Converter::readGraphFromGRFile(std::string filename, bool need_sort) {
  std::cout << "Reading " << filename << " graph into CPU memory\n";
  std::ifstream ifs;
  ifs.open(filename);
  int masterFD = open(filename.c_str(), O_RDONLY);
  if (masterFD == -1) {
    std::cout << "Graph: unable to open " << filename << "\n";
    exit(1);
  }
  struct stat buf;
  int f = fstat(masterFD, &buf);
  if (f == -1) {
    std::cout << "Graph: unable to stat " << filename << "\n";
    exit(1);
  }
  size_t masterLength = buf.st_size;
  int _MAP_BASE = MAP_PRIVATE;
  void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (m == MAP_FAILED) {
    m = 0;
    std::cout << "Graph: mmap failed.\n";
    exit(1);
  }
  Timer t;
  t.Start();
  uint64_t* fptr = (uint64_t*)m;
  __attribute__((unused)) uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  nv = le64toh(*fptr++);
  assert(nv < 2147483648); // assuming 31-bit (signed integer) vertex IDs
  ne = le64toh(*fptr++);
  uint64_t* outIdx = fptr;
  fptr += nv;
  uint32_t* fptr32 = (uint32_t*)fptr;
  uint32_t* outs = fptr32;
  fptr32 += ne;
  if (ne % 2) fptr32 += 1;
  if (sizeEdgeTy != 0) {
    std::cout << "sizeEdgeType = " << sizeEdgeTy << "\n";
    std::cout << "Graph: currently edge data not supported.\n";
    //exit(1);
  }
  std::cout << "constructing CSR: nv " << nv << " ne " << ne << "\n";
  g = new Graph(nv, ne);
  auto rowptr = g->out_rowptr();
  for (vidType vid = 0; vid < g->V(); ++vid) {
    g->fixEndEdge(vid, le64toh(outIdx[vid]));
    auto degree = rowptr[vid + 1] - rowptr[vid];
    for (unsigned jj = 0; jj < degree; ++jj) {
      eidType eid = rowptr[vid] + jj;
      vidType dst = le32toh(outs[eid]);
      if (dst >= g->V()) {
        printf("\tinvalid edge from %d to %d at index %d(%ld).\n", vid, dst, jj, eid);
        exit(0);
      }
      g->constructEdge(eid, dst);
    }
  }
  ifs.close();
  t.Stop();
  auto runtime = t.Seconds();
  std::cout << "read " << masterLength << " bytes in " << runtime << " sec ("
            << masterLength / 1000.0 / runtime << " MB/s)\n";

  t.Start();
  degrees.resize(nv);
  #pragma omp parallel for
  for (vidType v = 0; v < g->V(); v++) {
    degrees[v] = g->get_degree(v);
  }
  auto max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  std::cout << "max_degree: " << max_degree << "\n";

  if (need_sort) g->sort_neighbors();
  g->print_meta_data();
}

void Converter::read_labels(std::string filename, size_t num_classes, bool is_single_class) {
  Timer t_read;
  t_read.Start();
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  if (!is_single_class) {
    std::cout << "Using multi-class (multi-hot) labels\n";
    vlabels.resize(nv * num_classes); // multi-class label for each vertex: N x E
  } else {
    std::cout << "Using single-class (one-hot) labels\n";
    vlabels.resize(nv); // single-class (one-hot) label for each vertex: N x 1
  }
  std::cout << "Number of classes (unique label counts): " << num_classes;

  unsigned v = 0;
  while (std::getline(in, line)) {
    std::istringstream label_stream(line);
    unsigned x;
    for (size_t idx = 0; idx < num_classes; ++idx) {
      label_stream >> x;
      if (is_single_class) {
        if (x != 0) {
          vlabels[v] = idx;
          break;
        }
      } else {
        vlabels[v * num_classes + idx] = x;
      }
    }
    v++;
  }
  in.close();
  t_read.Stop();
  // print the number of vertex classes
  std::cout << ", time: " << t_read.Millisecs() << " ms\n";
}

size_t Converter::read_masks(std::string mask_type, std::string filename, size_t begin_, size_t end_, mask_t* masks) {
  size_t sample_count = 0, begin, end, i = 0;
  //std::ifstream f_in;
  //f_in.open(filename, std::ios::in);
  //f_in >> begin >> end >> std::ws;
  std::ifstream f_in(filename.c_str());
  assert(f_in);
  f_in >> begin >> end >> std::ws;
  std::cout << mask_type << "_mask range: [" << begin << ", " << end << ") ";
  assert(begin == begin_ && end == end_);
  std::string line;
  while (std::getline(f_in, line)) {
    std::istringstream mask_stream(line);
    if (i >= begin && i < end) {
      unsigned mask = 0;
      mask_stream >> mask;
      if (mask == 1) {
        masks[i] = 1;
        sample_count++;
      }
    }
    i++;
  }
  f_in.close();
  std::cout << "Number of valid samples: " << sample_count << " ("
            << (float)sample_count / (float)nv * (float)100 << "\%)\n";
  return sample_count;
}

