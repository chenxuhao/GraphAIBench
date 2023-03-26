#include "hybrid_compressor.hpp"
#include "codecfactory.h"
using namespace SIMDCompressionLib;

void hybrid_compressor::compress() {
  std::cout << "Start hybrid compressing: degree_threshold = " 
            << degree_threshold << " zeta_k = " << this->_zeta_k << "\n";
  Timer t;
  t.Start();
  pre_encoding();
  t.Stop();
  std::cout << "Pre-encoding time: " << t.Seconds() << "\n";

  this->edges_unary.clear();
  this->edges_unary.resize(g->V());
  this->edges_vbyte.clear();
  this->edges_vbyte.resize(g->V());
  t.Start();
  #pragma omp parallel for
  for (vidType i = 0; i < g->V(); i++) {
    encode_vertex(i);
  }
  t.Stop();
  std::cout << "Encoding time: " << t.Seconds() << "\n";
}


void hybrid_compressor::encode_vertex(const size_type v) {
  auto deg = g->get_degree(v);
  if (deg > degree_threshold) { // use VByte encoding
    encode_vbyte(v, edges_vbyte[v]);
  } else { // use unary encoding
    edges_unary[v].assign(g->N(v).begin(), g->N(v).end());
    append_gamma(edges_unary[v], deg);
    if (deg == 0) return;
    encode_unary(v, edges_unary[v]);
  }
}

void hybrid_compressor::encode_unary(vidType v, bits& bit_array) {
  auto adj = g->N(v);
  append_zeta(bit_array, int_2_nat(adj[0] - v));
  for (vidType i = 0; i < adj.size(); i++) {
    auto value = adj[i] - adj[i - 1] - 1;
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
  osizes[v] = static_cast<vidType>(outsize);
}

void hybrid_compressor::write_compressed_colidx(std::string out_prefix) {
  auto of_graph = fopen((out_prefix + ".edge.bin").c_str(), "w");
  if (of_graph == 0) {
    std::cout << "graph file cannot create!" << std::endl;
    abort();
  }
  std::vector<unsigned char> buf;
  unsigned char cur = 0;
  int bit_count = 0;

  Timer t;
  t.Start();
  for (vidType v = 0; v < g->V(); v++) {
    auto deg = g->get_degree(v);
    if (deg > degree_threshold) { // use VByte encoding
      if (fwrite(edges_vbyte[v].data(), sizeof(vidType) * osizes[v], 1, of_graph) != 1) {
        std::cerr << "aborting" << std::endl;
        fclose(of_graph);
        return;
      }
    } else { // use unary encoding
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
      if (fwrite(buf.data(), sizeof(unsigned char), buf.size(), of_graph) != 1) {
        std::cerr << "aborting" << std::endl;
        fclose(of_graph);
        return;
      }
    }
  }
  t.Stop();
  std::cout << "Writing compressed edges time: " << t.Seconds() << "\n";
}

void hybrid_compressor::write_compressed_rowptr(std::string out_prefix) {
}

void printusage() {
  cout << " Try ./compress -s name-of-scheme input.bin output.bin" << endl;
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
  if (optind + 1 >= argc) {
    printusage();
    return -1;
  }
  OutOfCoreGraph g(argv[optind]);
  g.print_meta_data();
  auto compressor = hybrid_compressor(&g, atoi(argv[optind + 1]), scheme);
  compressor.compress();
  std::cout << scheme << " generation completed.\n";
  return 0;
}
