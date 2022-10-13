// Copyright 2022 MIT
// Contact: Xuhao Chen <cxh@mit.edu>
#include "converter.h"
#include "scan.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

Converter::Converter(std::string file_type, std::string file_name, bool is_bipartite) {
  if (file_type == "gr" || file_type == "sgr" || file_type == "csgr") {
    readGraphFromGRFile(file_name, true);
  } else {
    if (file_type == "mtx") {
      read_mtx(file_name, is_bipartite);
      if (has_edge_weights) weighted_adjlist2CSR();
      else adjlist2CSR();
    } else if (file_type == "edges") {
      // plain edgelist format
      read_edgelist(file_name);
      edgelist2CSR();
    } else {
      read_lg(file_name);
      edgelist2CSR();
    }
  }
}

void Converter::adjlist2CSR() {
  printf("|V| %ld |E| %ld\n", nv, ne);
/*
  std::cout << "Printing adj_list: \n";
  for (vidType i = 0; i < nv; i++) {
    std::cout << "vertex " << i << ": degree = " 
      << adj_lists[i].size() << " edgelist = [ ";
    for (auto u : adj_lists[i]) {
      std::cout << u << " ";
    }
    std::cout << "]\n";
  }
  */
  g = new Graph(nv, ne);
  degrees.resize(nv);
  for (vidType i = 0; i < g->V(); i ++)
    degrees[i] = adj_lists[i].size();
  auto max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  std::cout << "maximum degree: " << max_degree << "\n";
  std::vector<eidType> offsets(nv+1);
  parallel_prefix_sum<vidType,eidType>(degrees, offsets.data());
  #pragma omp parallel for
  for (vidType v = 0; v < g->V(); ++v) {
    g->fixEndEdge(v, offsets[v+1]);
  }
  #pragma omp parallel for
  for (vidType v = 0; v < g->V(); ++v) {
    for (auto u : adj_lists[v]) {
      assert(u < g->V());
      eidType eid = offsets[v];
      g->constructEdge(eid, u);
      offsets[v]++;
    }
  }
  offsets.clear();
  //g->sort_neighbors();
  //g->print_graph();
}

void Converter::weighted_adjlist2CSR() {
  printf("|V| %ld |E| %ld\n", nv, ne);
  weights.resize(ne);
  g = new Graph(nv, ne);
  degrees.resize(nv);
  for (vidType i = 0; i < g->V(); i ++)
    degrees[i] = weighted_adj_lists[i].size();
  std::vector<eidType> offsets(nv+1);
  parallel_prefix_sum<vidType,eidType>(degrees, offsets.data());
  #pragma omp parallel for
  for (vidType v = 0; v < g->V(); ++v) {
    g->fixEndEdge(v, offsets[v+1]);
    for (auto neigh : weighted_adj_lists[v]) {
      auto u = neigh.first;
      auto label = neigh.second;
      assert(u < g->V());
      eidType eid = offsets[v];
      g->constructEdge(eid, u);
      weights[eid] = label;
      offsets[v]++;
    }
  }
  offsets.clear();
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

void Converter::edgelist2CSR() {
}

void Converter::weighted_edgelist2CSR() {
  WeightedEdgeList el(weighted_edge_set.begin(), weighted_edge_set.end());
  std::cout << "edgelist size: " << el.size() << "\n";
  ne = el.size(); // redundant edges removed
  printf("|V| %ld |E| %ld\n", nv, ne);

  // build CSR
  degrees.resize(nv);
  CountDegrees(el);
  g = new Graph(nv, ne);
  std::vector<eidType> offsets(nv+1);
  parallel_prefix_sum<vidType,eidType>(degrees, offsets.data());

  #pragma omp parallel for
  for (vidType v = 0; v < g->V(); ++v) {
    g->fixEndEdge(v, offsets[v+1]);
  }
  if (has_edge_weights) weights.resize(ne);
  for (auto e : el) {
    assert(e.src < g->V());
    assert(e.dst < g->V());
    eidType eid = offsets[e.src];
    if (has_edge_weights) weights[eid] = e.label;
    g->constructEdge(eid, e.dst);
    offsets[e.src]++;
  }
  offsets.clear();
  //g->print_graph();

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
  g->write_to_file(outfilename, v, e);
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

void Converter::CountDegrees(WeightedEdgeList el, bool symmetrize, bool transpose) {
  for (auto e : el) {
    if (symmetrize || (!symmetrize && !transpose))
      degrees[e.src] ++;
    if (symmetrize || (!symmetrize && transpose))
      degrees[e.dst] ++;
  }
  auto max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  std::cout << "maximum degree: " << max_degree << "\n";
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
    vidType src = atoi(result[0].c_str());
    vlabels.resize(src + 1);
    vlabels[src] = atoi(result[1].c_str());
    std::set<std::pair<vidType, OldEdgeValueT> > neighbors;
    for(size_t i = 2; i < result.size(); i++) {
      vidType dst = atoi(result[i].c_str());
      if (src == dst) continue; // remove self-loop
      neighbors.insert(std::pair<vidType, OldEdgeValueT>(dst, 0)); // remove redundant edge
    }
    for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
      WeightedEdge<OldEdgeValueT> edge(src, it->first, it->second);
      if (weighted_edge_set.find(edge) == weighted_edge_set.end()) {
        weighted_edge_set.insert(edge);
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
  uint64_t line_number = 0;
  while (infile.getline(line, 1024)) {
    result.clear();
    utils::split(line, result);
    assert(result.size() >= 2);
    // assuming vertex ID starts from 1 in the file
    // but we use vertex IDs starting from 0
    vidType src = atoi(result[0].c_str());
    vidType dst = atoi(result[1].c_str());
    if (src<1 || dst<1)
      std::cout << "line " << line_number << ": src=" << src << " dst=" << dst << "\n";
    src--;
    dst--;
    assert(src >=0);
    assert(dst >=0);
    if (src == dst) continue; // remove self-loop
    if (src+1 > num) num = src+1;
    if (dst+1 > num) num = dst+1;
    OldEdgeValueT elabel = 0;
    if (result.size() > 2) atoi(result[3].c_str());
    WeightedEdge<OldEdgeValueT> edge(src, dst, elabel);
    if (weighted_edge_set.find(edge) == weighted_edge_set.end()) {
      weighted_edge_set.insert(edge);
      weighted_edge_set.insert(WeightedEdge<OldEdgeValueT>(dst, src, elabel));
    }
    line_number++;
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
      WeightedEdge<OldEdgeValueT> edge(from, to, elabel);
      if (weighted_edge_set.find(edge) == weighted_edge_set.end()) {
        weighted_edge_set.insert(edge);
        weighted_edge_set.insert(WeightedEdge<OldEdgeValueT>(to, from, elabel));
      }
    }
  }
  nv = vlabels.size();
  ne = weighted_edge_set.size();
}

void Converter::read_mtx(std::string infile_name, bool is_bipartite) {
  std::cout << "Reading MTX file " << infile_name << "\n";
  std::ifstream infile;
  infile.open(infile_name, std::ios::in);
  std::string start, object, format, field, symmetry, line;
  bool read_weights;
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
    has_edge_weights = false;
    std::cout << "This graph does not have edge weights\n";
  } else if ((field == "real") || (field == "double") || (field == "integer")) {
    read_weights = true;
    has_edge_weights = true;
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
  if (read_weights)
    weighted_adj_lists.resize(nv);
  else
    adj_lists.resize(nv);
  eidType idx = 0;
  int num_redundant_edges = 0;
  ne = 0;
  while (std::getline(infile, line)) {
    std::istringstream edge_stream(line);
    vidType u, v;
    edge_stream >> u;
    edge_stream >> v;
    assert(u > 0);
    assert(v > 0);
    auto src = u-1;
    auto dst = v-1;
    if (is_bipartite) dst += m;
    if (src == dst) continue; // remove selfloops
    if (read_weights) {
      OldEdgeValueT elabel = 0;
      edge_stream >> elabel;
      Neighbor neigh(std::make_pair(dst, elabel));
      if (weighted_adj_lists[src].find(neigh) == weighted_adj_lists[src].end()) { // remove redundant edges
        weighted_adj_lists[src].insert(neigh);
        ne ++;
        if (undirected) {
          Neighbor reverse_neigh(std::make_pair(src, elabel));
          weighted_adj_lists[dst].insert(reverse_neigh);
          ne ++;
        }
      } else {
        num_redundant_edges ++;
      }
    } else {
      if (adj_lists[src].find(dst) == adj_lists[src].end()) { // remove redundant edges
        adj_lists[src].insert(dst);
        ne ++;
        if (undirected) {
          adj_lists[dst].insert(src);
          ne ++;
        }
      } else {
        num_redundant_edges ++;
      }
    }
    idx ++;
    if (idx%1000000000 == 0)
      std::cout << "Finish reading " << idx << " lines so far\n";
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
  #pragma omp parallel for
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

void Converter::splitGRFile(std::string filename, std::string outfilename) {
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
  // write rowptr
  eidType head = 0;
  std::cout << "converting GR file: nv " << nv << " ne " << ne << "\n";
  std::cout << "rowptr[ " << nv-1 << "] = " << outIdx[nv-1] << "\n";
  ///*
  if (nv) {
    std::ofstream outfile((outfilename+".vertex.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(&head), sizeof(eidType));
    outfile.write(reinterpret_cast<const char*>(outIdx), nv*sizeof(eidType));
    outfile.close();
  }
  //*/
  ///*
  if (ne) {
    std::ofstream outfile1((outfilename+".edge.bin").c_str(), std::ios::binary);
    if (!outfile1) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile1.write(reinterpret_cast<const char*>(outs), ne*sizeof(vidType));
    outfile1.close();
  }
  //*/
}

