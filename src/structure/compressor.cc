#include "compressor.hh"
#include "codecfactory.h"
#include "scan.h"
using namespace SIMDCompressionLib;

void Compressor::write_compressed_graph() {
  if (use_unary) {
    std::cout << "writing the compressed edges to disk\n";
    write_compressed_edges_to_disk();
  }
  std::cout << "Computing the row pointers\n";
  compute_ptrs();
  std::cout << "Writing the row pointers to disk\n";
  write_ptrs_to_disk();
}

void Compressor::compute_ptrs() {
  Timer t;
  t.Start();
  rowptr.resize(g->V()+1);
#if 0
  buffer.resize(g->V());
  #pragma omp parallel for
  for (vidType i = 0; i < g->V(); i++)
    buffer[i] = encoder->get_compressed_size(i).size();
  parallel_prefix_sum<vidType,eidType>(degrees, rowptr.data());
#else
  if (use_unary) {
    rowptr[0] = 0;
    for (vidType i = 0; i < g->V(); i++) {
      auto length = encoder->get_compressed_size(i);
      rowptr[i+1] = length + rowptr[i];
    }
  } else {
    parallel_prefix_sum<vidType,eidType>(osizes, rowptr.data());
  }
#endif
  t.Stop();
  std::cout << "Computing row pointers time: " << t.Seconds() << "\n";
}

void Compressor::write_ptrs_to_disk() {
  Timer t;
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

void Compressor::write_compressed_edges_to_disk() {
  std::string edge_file_name = out_prefix + ".edge.bin";
  FILE *of_graph = fopen((edge_file_name).c_str(), "w");
  if (of_graph == 0) {
    std::cout << "graph file " << edge_file_name << " cannot be created!\n";
    abort();
  }

  Timer t;
  t.Start();
  std::vector<unsigned char> buf;
  unsigned char cur = 0;
  int bit_count = 0;
#if 1
  for (vidType i = 0; i < g->V(); i++) {
    for (auto bit : encoder->get_compressed_bits(i)) {
      cur <<= 1;
      if (bit) cur++;
      bit_count++;
      if (bit_count == 8) {
        buf.emplace_back(cur);
        cur = 0;
        bit_count = 0;
      }
    }
  }
  if (bit_count) {
    while (bit_count < 8) cur <<= 1, bit_count++;
    buf.emplace_back(cur);
  }
#else
#endif
  fwrite(buf.data(), sizeof(unsigned char), buf.size(), of_graph);
  t.Stop();
  std::cout << "Time writing compressed edges to disk: " << t.Seconds() << "\n";
  fclose(of_graph);
}

void Compressor::compress(bool pre_encode) {
  Timer t;
  if (use_unary && pre_encode) {
    std::cout << "Pre-encoding ...\n";
    t.Start();
    encoder->pre_encoding();
    t.Stop();
    std::cout << "Pre-encoding time: " << t.Seconds() << "\n";
  }

  t.Start();
  if (use_unary) {
    #pragma omp parallel for
    for (vidType i = 0; i < g->V(); i++) {
      encoder->encode(i, g->get_degree(i), g->N(i).data());
    }
    encoder->print_stats();
  } else {
    FILE *of_graph = fopen((out_prefix + ".edge.bin").c_str(), "w");
    if (of_graph == 0) {
      std::cout << "graph file cannot create!" << std::endl;
      abort();
    }
    osizes.resize(g->V());
    for (vidType v = 0; v < g->V(); v++) {
      auto deg = g->get_degree(v);
      if (buffer.size() < deg + 1024) {
        buffer.resize(deg + 1024);
      }
      size_t outsize = buffer.size();
      shared_ptr<IntegerCODEC> schemeptr = CODECFactory::getFromName(scheme);
      if (schemeptr.get() == NULL) exit(1);
      schemeptr->encodeArray(g->adj_ptr(v), deg, buffer.data(), outsize);
      osizes[v] = static_cast<vidType>(outsize);
      if (fwrite(buffer.data(), sizeof(vidType) * outsize, 1, of_graph) != 1) {
        std::cerr << "aborting" << std::endl;
        fclose(of_graph);
        exit(1);
      }
    }
    fclose(of_graph);
  }
  t.Stop();
  std::cout << "Encoding time: " << t.Seconds() << "\n";
}

void printusage() {
  cout << "./compressor -s name-of-scheme <input_path> <output_path> [zeta_k] [use_interval]\n";
}

int main(int argc,char *argv[]) {
  std::string scheme = "cgr";
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

  int zeta_k = 3, use_interval = 1, add_degree = 0;
  if (argc > 5) zeta_k = atoi(argv[optind+2]);
  if (argc > 6) use_interval = atoi(argv[optind+3]);
  if (argc > 7) add_degree = atoi(argv[optind+4]);
  bool use_unary = (scheme == "cgr" || scheme == "hybrid");
  std::cout << "Using the " << scheme << " compression scheme\n";

  unary_encoder *encoder = NULL;
  if (use_unary) {
    std::cout << "Creating a CGR encoder\n";
    encoder = new cgr_encoder(g.V(), zeta_k, use_interval, add_degree);
  }
  Compressor compressor(scheme, argv[optind+1], use_unary, &g, encoder);
  std::cout << "start compression ...\n";
  compressor.compress();
  std::cout << "writing compressed graph to disk ...\n";
  compressor.write_compressed_graph();
  std::cout << "compression completed!\n";
  return 0;
}
