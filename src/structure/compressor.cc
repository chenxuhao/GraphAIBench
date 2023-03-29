#include "compressor.hh"

void Compressor::write_compressed_graph() {
  write_compressed_edges_to_disk();
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
  rowptr[0] = 0;
  for (vidType i = 0; i < g->V(); i++)
    rowptr[i+1] = encoder->get_compressed_size(i) + rowptr[i];
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
  std::cout << "writing the compressed edges to disk\n";
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
  std::cout << "Pre-encoding ...\n";
  Timer t;
  t.Start();
  if (pre_encode) encoder->pre_encoding();
  t.Stop();
  std::cout << "Pre-encoding time: " << t.Seconds() << "\n";

  t.Start();
  #pragma omp parallel for
  for (vidType i = 0; i < g->V(); i++) {
    encoder->encode(i, g->get_degree(i), g->N(i).data());
  }
  t.Stop();
  std::cout << "Encoding time: " << t.Seconds() << "\n";
  encoder->print_stats();
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

  unary_encoder *encoder = new cgr_encoder(g.V(), zeta_k, use_interval, add_degree);
  Compressor compressor(scheme, argv[optind+1], &g, encoder);
  std::cout << "start compression ...\n";
  compressor.compress();
  std::cout << "writing compressed graph to disk ...\n";
  compressor.write_compressed_graph();
  std::cout << "compression completed!\n";
  return 0;
}
