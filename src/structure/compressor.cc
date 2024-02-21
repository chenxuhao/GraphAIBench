#include "compressor.hh"
#include "scan.h"
#include "cgr_encoder.hh"
#include "vbyte_encoder.hh"
#include <fstream>
//#include "hybrid_encoder.hh"

#define CHECKPOINT 50000000


void copy_meta_file(std::string in_prefix, std::string out_prefix) {
    std::ifstream  src(in_prefix + ".meta.txt", std::ios::binary);
    std::ofstream  dst(out_prefix + ".meta.txt", std::ios::binary);
    dst << src.rdbuf();
}


void Compressor::write_compressed_graph() {
  if (scheme == "cgr")
    write_compressed_edges_to_disk();
  std::cout << "Computing the row pointers\n";
  compute_ptrs();
  write_ptrs_to_disk();
}

void Compressor::compute_ptrs() {
  Timer t;
  t.Start();
  rowptr.resize(g->V()+1);
  rowptr[0] = 0;
  // if (scheme == "cgr" && !word_aligned) {
  if (!word_aligned) {
    assert (scheme == "cgr");
    #pragma omp parallel for
    for (vidType i = 0; i < g->V(); i++) {
      auto length = encoder->get_compressed_bits_size(i);
      if (byte_aligned && length > 0)
        length = ((length-1)/8 + 1);
      if (word_aligned && length > 0)
        length = ((length-1)/32 + 1);
      osizes[i] = length;
    }
  }
  parallel_prefix_sum<vidType,eidType>(osizes, rowptr.data());
  t.Stop();
  std::cout << "Computing row pointers time: " << t.Seconds() << "\n";
}

void Compressor::write_ptrs_to_disk() {
  std::string filename = out_prefix + ".vertex.bin";
  std::cout << "Writing the row pointers to disk file " << filename << "\n";
  Timer t;
  t.Start();
  std::ofstream outfile((filename).c_str(), std::ios::binary);
  if (!outfile) {
    std::cout << "File not available\n";
    throw 1;
  }
  outfile.write(reinterpret_cast<const char*>(rowptr.data()), (g->V()+1)*sizeof(eidType));
  outfile.close();
  t.Stop();
  std::cout << "Writing row pointers time: " << t.Seconds() << "\n";
}

void Compressor::bits_to_bytes(bits bit_array, std::vector<unsigned char> &buf, unsigned char &cur, int &bit_count) {
  int byte_count = 0, res_byte_count = 0;
  if (bit_array.size() == 0) return;
  byte_count = (bit_array.size() - 1) / 8 + 1;
  res_byte_count = byte_count % 4;
  for (auto bit : bit_array) {
    cur <<= 1;
    if (bit) cur++;
    bit_count++;
    if (bit_count == 8) {
      buf.emplace_back(cur);
      cur = 0;
      bit_count = 0;
    }
  }
  if (bit_count && (word_aligned || byte_aligned)) {
    // byte alignment for each adj list
    while (bit_count < 8) cur <<= 1, bit_count++;
    buf.emplace_back(cur);
    bit_count = 0;
    cur = 0;
  }
  if (word_aligned && res_byte_count) {
    while (res_byte_count != 4) {
      cur = 0; // padding
      buf.emplace_back(cur);
      res_byte_count ++;
    }
  }
}

void Compressor::write_compressed_edges_to_disk() {
  std::string edge_file_name = out_prefix + ".edge.bin";
  std::cout << "writing the compressed edges to disk file " << edge_file_name << "\n";
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
  for (vidType i = 0; i < g->V(); i++) {
    auto &bit_array = encoder->get_compressed_bits(i);
    bits_to_bytes(bit_array, buf, cur, bit_count);
  }
  // the last byte; only used when not byte aligned or word aligned
  if (bit_count) {
    while (bit_count < 8) cur <<= 1, bit_count++;
    buf.emplace_back(cur);
  }
  if (word_aligned && use_permutate)
    permutate_bytes_by_word(buf);
  fwrite(buf.data(), sizeof(unsigned char), buf.size(), of_graph);
  t.Stop();
  std::cout << "Time writing compressed edges to disk: " << t.Seconds() << "\n";
  fclose(of_graph);
}

void Compressor::permutate_bytes_by_word(std::vector<unsigned char> &buf) {
  auto num_bytes = buf.size();
  auto res_bytes = num_bytes % 4;
  assert(res_bytes == 0); // only does this when word aligned
  auto num_words = (num_bytes-res_bytes)/4 + (res_bytes?1:0);
  //#pragma omp parallel for
  for (size_t i = 0; i < num_words; i++) {
    auto j = i*4;
    auto temp = buf[j];
    buf[j] = buf[j+3];
    buf[j+3] = temp;
    temp = buf[j+1];
    buf[j+1] = buf[j+2];
    buf[j+2] = temp;
  }
}

void Compressor::write_degrees() {
  Timer t;
  t.Start();
  std::string degree_filename = (out_prefix + ".degree.bin").c_str();
  std::cout << "Writing degrees to disk " << degree_filename << "\n";
  std::ofstream outfile(degree_filename, std::ios::binary);
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

void Compressor::compress(bool pre_encode, bool reverse) {
  if (byte_aligned) std::cout << "Byte alignment enabled for each adj list\n";
  if (word_aligned) std::cout << "Word alignment enabled for each adj list\n";
  Timer t;
  if (use_unary && pre_encode) {
    std::cout << "Pre-encoding ...\n";
    t.Start();
    encoder->pre_encoding();
    t.Stop();
    std::cout << "Pre-encoding time: " << t.Seconds() << "\n";
  }
  osizes.resize(g->V());
  vbyte_count = 0, unary_count = 0, trivial_count = 0;
  vbyte_adj_count = 0, unary_adj_count = 0;
  unary_bytes = 0, vbyte_bytes = 0;

  std::string filename = out_prefix + ".edge.bin";
  FILE *of_graph = fopen(filename.c_str(), "w");
  if (of_graph == 0) {
    std::cout << "graph file cannot create!" << std::endl;
    exit(1);
  }
  std::string vbyte_scheme = "streamvbyte";
  vbyte_encoder vb_encoder(vbyte_scheme);

  std::cout << "Start encoding\n"; 
  t.Start();
  //#pragma omp parallel for
  //#pragma omp parallel for reduction(+:vbyte_count,unary_count,trivial_count,unary_bytes,vbyte_bytes) schedule(dynamic, 1)
  for (vidType v = 0; v < g->V(); v++) {
    if (v > 0 && v%CHECKPOINT==0)
      std::cout << "(" << v/CHECKPOINT << " * " << CHECKPOINT << ") vertices compressed\n";
    auto deg = g->get_degree(v);
    if (deg == 0) trivial_count ++;
    bool do_vbyte = !use_unary || (scheme == "hybrid" && (reverse? deg <= degree_threshold : deg > degree_threshold));

    // encode the neighbor list
    if (do_vbyte) {
      if (buffer.size() < deg + 1024) buffer.resize(deg + 1024);
      osizes[v] = vb_encoder.encode(deg, g->N(v).data(), buffer.data(), scheme != "hybrid");
    } else { // unary encoding
      osizes[v] = encoder->encode(v, deg, g->N(v).data());
    }

    // write to disk
    if (do_vbyte) {
      if (fwrite(buffer.data(), sizeof(vidType) * osizes[v], 1, of_graph) != 1) {
        std::cerr << "[vbyte] write file " << filename << " failed: aborting\n";
        fclose(of_graph);
        exit(1);
      }
    } else if (scheme == "hybrid") {
      std::vector<unsigned char> buf;
      unsigned char cur = 0;
      int bit_count = 0;
      auto &bit_array = encoder->get_compressed_bits(v);
      bits_to_bytes(bit_array, buf, cur, bit_count);
      assert(bit_count == 0);
      assert(cur == 0);
      if (word_aligned && use_permutate)
        permutate_bytes_by_word(buf);
      if (fwrite(buf.data(), sizeof(unsigned char), buf.size(), of_graph) != buf.size()) {
        std::cerr << "[unary] write file " << filename << " failed: aborting\n";
        fclose(of_graph);
        exit(1);
      }
    }
 
    // update statistics
    if (do_vbyte) { // use vbyte encoding
      vbyte_count ++;
      vbyte_adj_count += deg;
      vbyte_bytes += osizes[v] * 4;
    } else { // use unary encoding
      unary_count ++;
      unary_adj_count += deg;
      unary_bytes += osizes[v] * 4;
    }
  }
  //if (use_unary) encoder->print_stats();
  fclose(of_graph);
  t.Stop();
  std::cout << "Encoding time: " << t.Seconds() << "\n";
}

void Compressor::print_stats() {
  std::cout << "vbyte_count: " << vbyte_count << " unary_count: " << unary_count << " trivial_count: " << trivial_count << "\n";
  float vbyte_rate = float(vbyte_adj_count)*4.0/float(vbyte_bytes);
  float unary_rate = float(unary_adj_count)*4.0/float(unary_bytes);
  std::cout << "VByte bytes: " << float(vbyte_bytes)/1024/1024 
            << " MB, original data: " << float(vbyte_adj_count)*4.0/1024/1024 
            << " MB, compression rate: " << vbyte_rate << "\n";
  std::cout << "Unary bytes: " << float(unary_bytes)/1024/1024 
            << " MB, original data: " << float(unary_adj_count)*4.0/1024/1024 
            << " MB, compression rate: " << unary_rate << "\n";
}

void printusage() {
  cout << "./compressor -s name-of-scheme <input_path> <output_path> [-z zeta_k(3)]"
       <<                                                          " [-i use_interval]"
       <<                                                          " [-r residual_segment_length]"
       <<                                                          " [-p permutate_bytes]"
       <<                                                          " [-d degree_threshold(32)]"
       <<                                                          " [-a alignment(0)]\n";
}

// int main(int argc,char *argv[]) {
//   int zeta_k = 2, permutate = 0, degree_threshold = 32;
//   int alignment = 0; // 0: not aligned; 1: byte aligned; 2: word aligned
//   bool reverse = false; // reverse hybrid scheme: low-degree vbyte; high-degree unary
//   bool use_interval = false; // use interval for CGR format
//   bool use_segment = false;
//   std::string scheme = "cgr";
//   int c;
//   while ((c = getopt(argc, argv, "s:z:igrpa:d:h")) != -1) {
//     switch (c) {
//       case 's':
//         scheme = optarg;
//         break;
//       case 'z':
//         zeta_k = atoi(optarg);
//         break;
//       case 'i':
//         use_interval = true;
//         break;
//       case 'g':
//         use_segment = true;
//         break;
//       case 'r':
//         reverse = true;
//         break;
//       case 'p':
//         permutate = 1;
//         break;
//       case 'a':
//         alignment = atoi(optarg);
//         break;
//       case 'd':
//         degree_threshold = atoi(optarg);
//         break;
//       case 'h':
//         printusage();
//         return 0;
//       case '?': 
//         printf("unknown option: %c\n", optopt);
//         break;
//       default:
//         abort();
//     }
//   }
//   if (optind + 1 >= argc) {
//     printusage();
//     return -1;
//   }
//   if (use_interval && !use_segment) {
//     printf("Error: non-segmented interval is currently not supported\n");
//     exit(1);
//   }
 
//   GraphTy g(argv[optind]);
//   g.print_meta_data();

//   bool use_unary = (scheme == "cgr" || scheme == "hybrid");
//   std::cout << "Using the " << scheme << " compression scheme\n";
//   if (scheme == "streamvbyte") {
//     if (alignment != 2) {
//       std::cout << "vbyte scheme must be word-aligned\n";
//       exit(1);
//     }
//   }
//   if (scheme == "hybrid") {
//     if (!permutate) {
//       std::cout << "hybrid scheme must be permutated\n";
//       exit(1);
//     }
//     if (alignment != 2) {
//       std::cout << "hybrid scheme must be word-aligned\n";
//       exit(1);
//     }
//     std::cout << "degree_threshold = " << degree_threshold << "\n";
//   }

//   bool pre_encode = g.V() > 1000000;
//   unary_encoder *encoder = NULL;
//   if (use_unary)
//     encoder = new cgr_encoder(g.V(), zeta_k, pre_encode, use_interval, use_segment);
//   Compressor compressor(scheme, argv[optind+1], use_unary, &g, encoder, permutate, degree_threshold, alignment);
//   std::cout << "start compression ...\n";
//   compressor.compress(pre_encode, reverse);
//   compressor.print_stats();
//   std::cout << "writing compressed graph to disk ...\n";
//   compressor.write_compressed_graph();
//   if (scheme == "hybrid") {
//     std::cout << "Start writing degrees (hybrid scheme only) to disk ...\n";
//     compressor.write_degrees();
//   } 
//   std::cout << "compression completed!\n";
//   copy_meta_file(argv[optind], argv[optind+1]);
//   std::cout << "meta file copied over\n";
//   return 0;
// }
