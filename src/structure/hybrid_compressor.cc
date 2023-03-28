#include "hybrid_compressor.hpp"
#include "codecfactory.h"
using namespace SIMDCompressionLib;
#define CHECKPOINT 50000000

void hybrid_compressor::compress() {
  std::cout << "Start hybrid compressing: degree_threshold = " 
            << degree_threshold << " zeta_k = " << this->_zeta_k << "\n";
  Timer t;
  t.Start();
  pre_encoding();
  t.Stop();
  std::cout << "Pre-encoding time: " << t.Seconds() << "\n";

  edges_unary.resize(g->V());
  edges_vbyte.resize(g->V());
  osizes.resize(g->V());
  t.Start();
  vidType vbyte_count = 0, unary_count = 0, trivial_count = 0;
  vidType vbyte_adj_count = 0, unary_adj_count = 0;
  int64_t unary_bytes = 0, vbyte_bytes = 0;
  std::cout << "Start encoding\n"; 
  #pragma omp parallel for reduction(+:vbyte_count,unary_count,trivial_count,unary_bytes,vbyte_bytes) schedule(dynamic, 1)
  for (vidType v = 0; v < g->V(); v++) {
    if (v > 0 && v%CHECKPOINT==0) std::cout << "(" << v/CHECKPOINT << " * " << CHECKPOINT << ") vertices compressed\n";
    auto deg = g->get_degree(v);
    if (deg == 0) {
      trivial_count ++;
      continue;
    }
    //std::cout << "v=" << v << ", deg=" << deg << "\n";
    if (deg > degree_threshold) { // use VByte encoding
      edges_vbyte[v].clear();
      encode_vbyte(v, edges_vbyte[v]);
      vbyte_count ++;
      vbyte_adj_count += deg;
      vbyte_bytes += osizes[v] * 4;
    } else { // use unary encoding
      edges_unary[v].clear();
      encode_unary(v, edges_unary[v]);
      assert(edges_unary[v].size() > 0);
      osizes[v] = (edges_unary[v].size() - 1)/8 + 1; // number of bits --> number of bytes
      unary_count ++;
      unary_adj_count += deg;
      unary_bytes += osizes[v];
    }
  }
  t.Stop();
  std::cout << "Encoding time: " << t.Seconds() << "\n";
  std::cout << "vbyte_count: " << vbyte_count << " unary_count: " << unary_count << " trivial_count: " << trivial_count << "\n";
  float vbyte_rate = float(vbyte_adj_count)*4.0/float(vbyte_bytes);
  float unary_rate = float(unary_adj_count)*4.0/float(unary_bytes);
  std::cout << "VByte bytes: " << float(vbyte_bytes)/1024/1024 << " MB, compression rate: " << vbyte_rate << "\n";
  std::cout << "Unary bytes: " << float(unary_bytes)/1024/1024 << " MB, compression rate: " << unary_rate << "\n";
}

void hybrid_compressor::encode_vertex(const size_type v) {
}

void hybrid_compressor::encode_unary(vidType v, bits& bit_array) {
  auto adj = g->N(v);
  auto degree = g->get_degree(v);
  //if (v == 185) std::cout << "adj[0]=" << adj[0] << "\n";
  int64_t value = int_2_nat(int64_t(adj[0]) - int64_t(v));
  //if (v == 185) std::cout << "length=" << bit_array.size() << ", value=" << value << "\n";
  append_zeta(bit_array, value);
  for (vidType i = 1; i < degree; i++) {
    value = int64_t(adj[i]) - int64_t(adj[i - 1]) - 1;
    //if (v == 185) std::cout << "adj[" << i << "]=" << adj[i] << ", adj[" << i-1 << "]=" << adj[i-1] << ", value=" << value << "\n";
    append_zeta(bit_array, value);
  }
}

void hybrid_compressor::encode_vbyte(vidType v, std::vector<uint32_t>& buffer) {
  auto deg = g->get_degree(v);
  size_t outsize = deg + 1024;
  if (buffer.size() < outsize) buffer.resize(outsize);
  outsize = buffer.size();
  shared_ptr<IntegerCODEC> schemeptr = CODECFactory::getFromName(vbyte_scheme);
  if (schemeptr.get() == NULL) return;
  schemeptr->encodeArray(g->adj_ptr(v), deg, buffer.data(), outsize);
  osizes[v] = static_cast<vidType>(outsize); // number of words
}

void hybrid_compressor::write_compressed_colidx(std::string out_prefix) {
  auto of_graph = fopen((out_prefix + ".edge.bin").c_str(), "w");
  if (of_graph == 0) {
    std::cout << "graph file cannot create!" << std::endl;
    abort();
  }

  Timer t;
  t.Start();
  for (vidType v = 0; v < g->V(); v++) {
    auto deg = g->get_degree(v);
    //std::cout << "writing v=" << v << ", degree=" << deg << "\n";
    if (deg > degree_threshold) { // use VByte encoding
      if (fwrite(edges_vbyte[v].data(), sizeof(uint32_t), osizes[v], of_graph) != osizes[v]) {
        perror("Could not write to file");
        exit(1);
        //printf("Number of items written to the file: %ld\n", num);
      }
      osizes[v] = 4 * osizes[v]; // number of words --> number of bytes
    } else { // use unary encoding
      int bit_count = 0;
      unsigned char cur = 0;
      std::vector<unsigned char> buf;
      buf.clear();
      for (auto bit : edges_unary[v]) {
        cur <<= 1;
        if (bit) cur++;
        bit_count++;
        if (bit_count == 8) {
          buf.emplace_back(cur);
          cur = 0;
          bit_count = 0;
        }
      }
      if (bit_count) {
        while (bit_count < 8) cur <<= 1, bit_count++;
        buf.emplace_back(cur);
      }
      if (fwrite(buf.data(), sizeof(unsigned char), buf.size(), of_graph) != buf.size()) {
        perror("Could not write to file");
        exit(1);
      }
    }
  }
  if (fclose(of_graph) != 0) {
    //Exception Handling if file could not be closed
    perror("Could not close file");
    exit(1);
  }
  t.Stop();
  std::cout << "Writing compressed edges time: " << t.Seconds() << "\n";
}

void hybrid_compressor::write_compressed_rowptr(std::string out_prefix) {
  std::cout << "Computing the row pointers\n";
  Timer t;
  t.Start();
  std::vector<eidType> rowptr(g->V()+1);
#if 0
  for (vidType i = 0; i < g->V(); i++)
    degrees[i] = osizes.size();
  parallel_prefix_sum<vidType,eidType>(osizes.data(), rowptr.data());
#else
  rowptr[0] = 0;
  for (vidType i = 0; i < g->V(); i++)
    rowptr[i+1] = osizes[i] + rowptr[i]; // byte offsets
#endif
  t.Stop();
  std::cout << "Computing row pointers time: " << t.Seconds() << "\n";

  std::cout << "Writing the row pointers to disk\n";
  t.Start();
  std::ofstream outfile((out_prefix + ".vertex.bin").c_str(), std::ios::binary);
  if (!outfile) {
    std::cout << "File not available\n";
    throw 1;
  }
  outfile.write(reinterpret_cast<const char*>(rowptr.data()), (g->V()+1)*sizeof(eidType));
  outfile.close();
  t.Stop();
  std::cout << "Writing row pointers time: " << t.Seconds() << "\n";
}

void hybrid_compressor::write_degrees(std::string out_prefix) {
  std::cout << "Writing degrees to disk\n";
  Timer t;
  t.Start();
  std::ofstream outfile((out_prefix + ".degree.bin").c_str(), std::ios::binary);
  if (!outfile) {
    std::cout << "File not available\n";
    throw 1;
  }
  std::vector<vidType> degrees(g->V());
  for (vidType v = 0; v < g->V(); v++) degrees[v] = g->get_degree(v);
  outfile.write(reinterpret_cast<const char*>(degrees.data()), (g->V())*sizeof(vidType));
  outfile.close();
  t.Stop();
  std::cout << "Writing degrees time: " << t.Seconds() << "\n";
}

void printusage() {
  cout << " Try ./hybrid_compressor -s name-of-scheme in_prefix out_prefix [zeta_k(2)] [degree_threshold(32)]" << endl;
}

int main(int argc,char *argv[]) {
  std::string scheme;
  int c;
  while ((c = getopt(argc, argv, "s:h")) != -1) {
    switch (c) {
      case 's':
        scheme = optarg;
        break;
      case 'h':
        printusage();
        return 0;
      default:
        abort();
    }
  }
  if (argc < 3) {
    printusage();
    return -1;
  }
  std::string infile = argv[optind];
  std::string outfile = argv[optind+1];
  OutOfCoreGraph g(infile);
  g.print_meta_data();
  int zeta_k = 2;
  if (argc > 5) zeta_k = atoi(argv[optind + 2]);
  vidType deg = 32;
  if (argc > 6) deg = atoi(argv[optind + 3]);
  //std::cout << "zeta_k = " << zeta_k << ", degree_threshold = " << deg << "\n";

  auto compressor = hybrid_compressor(&g, zeta_k, deg, scheme);
  compressor.compress();
  std::cout << "Start writing degrees to disk ...\n";
  compressor.write_degrees(outfile);
  std::cout << "Start writing edges (colidx) to disk ...\n";
  compressor.write_compressed_colidx(outfile);
  std::cout << "Start writing vertices (rowptr) to disk ...\n";
  compressor.write_compressed_rowptr(outfile);
  std::cout << scheme << " generation completed.\n";
  return 0;
}
