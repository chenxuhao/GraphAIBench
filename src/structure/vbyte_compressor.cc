#include "vbyte_compressor.hpp"
#include "codecfactory.h"
using namespace SIMDCompressionLib;
// split the graph into two parts: high-degree vertices and low-degree vertices
// assuming the vertex ids are sorted by degree
/*
void vbyte_compressor::split_high_low(vidType v) {
  for (vidType v = 0; v < g->V(); v++) {
    auto deg = g->get_degree(v);
    if (deg >= 32) {
    }
  }
}
*/
void vbyte_compressor::compress() {
  std::cout << "Start compressing \n";
  of_graph = fopen((out_prefix + ".edge.bin").c_str(), "w");
  if (of_graph == 0) {
    std::cout << "graph file cannot create!" << std::endl;
    abort();
  }
  osizes.resize(g->V());
  Timer t;
  t.Start();
  for (vidType v = 0; v < g->V(); v++) {
    encode_vertex(v);
  }
  t.Stop();
  fclose(of_graph);
  write_ptrs_to_disk();
}

int vbyte_compressor::encode_vertex(vidType v) {
  auto deg = g->get_degree(v);
  //std::cout << "Encoding vertex " << v << " with degree " << deg << "\n";
  if (buffer.size() < deg + 1024) {
    buffer.resize(deg + 1024);
  }
  size_t outsize = buffer.size();
  shared_ptr<IntegerCODEC> schemeptr = CODECFactory::getFromName(scheme);
  if (schemeptr.get() == NULL) return -2;
  schemeptr->encodeArray(g->adj_ptr(v), deg, buffer.data(), outsize);
  osizes[v] = static_cast<vidType>(outsize);
  if (fwrite(buffer.data(), sizeof(vidType) * outsize, 1, of_graph) != 1) {
    std::cerr << "aborting" << std::endl;
    fclose(of_graph);
    return -1;
  }
  return 0;
}

void vbyte_compressor::write_ptrs_to_disk() {
  std::cout << "Computing the row pointers\n";
  Timer t;
  t.Start();
  std::vector<eidType> rowptr(g->V()+1);
#if 0
  parallel_prefix_sum<vidType,eidType>(osizes, rowptr.data());
#else
  rowptr[0] = 0;
  for (vidType i = 0; i < g->V(); i++)
    rowptr[i+1] = osizes[i] + rowptr[i];
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
  auto compressor = vbyte_compressor(scheme, argv[optind + 1], &g);
  compressor.compress();
  std::cout << scheme << " generation completed.\n";
  return 0;
}
