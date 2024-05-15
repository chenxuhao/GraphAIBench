#include "graph.h"
#include <bits/stdc++.h>
#include <cmath>
#include <fstream>
#include "vbyte_encoder.hh"
#include <string>
#include <vector>
using namespace std;

void write_to_file(std::string outfilename, size_t ne, size_t nv, vidType max_deg, eidType *rowptrs, vidType *colidxs) {
  std::cout << "Writing graph to file\n";
  std::ofstream outfile((outfilename+".vertex.bin").c_str(), std::ios::binary);
  if (!outfile) {
    std::cout << "File not available\n";
    throw 1;
  }
  outfile.write(reinterpret_cast<const char*>(rowptrs), (nv+1)*sizeof(eidType));
  outfile.close();

  std::ofstream outfile1((outfilename+".edge.bin").c_str(), std::ios::binary);
  if (!outfile1) {
    std::cout << "File not available\n";
    throw 1;
  }
  outfile1.write(reinterpret_cast<const char*>(colidxs), (ne)*sizeof(vidType));
  outfile1.close();

  std::ofstream outfile2((outfilename+".meta.txt").c_str(), std::ios::binary);
  if (!outfile1) {
    std::cout << "File not available\n";
    throw 1;
  }
  std::string meta = std::to_string(nv) + "\n" + std::to_string(ne) + "\n";
  meta += "4 8 1 4\n";
  meta += std::to_string(max_deg) + "\n";
  meta += "0\n0\n0\n";
  outfile2 << meta;
  outfile2.close();
  std::cout << "File name " << outfilename << std::endl;
}

void build_uncomp(Graph &g, vidType nv, eidType ne, std::string out_file) {
  if (ne == 0) return;
  vidType *_edges = new vidType[ne];
  vidType *_buff = _edges;
  eidType *_vertices = new eidType[nv + 1];
  _vertices[0] = 0;
  std::cout << "nv " << nv << " ne " << ne << std::endl;
  for (vidType i = 0; i < nv; i++) {	    
    vidType deg = g.decode_vertex_vbyte(i, _edges, "streamvbyte");
    _edges += deg;
    _vertices[i+1] = deg + _vertices[i];
  }
  _edges -= ne;
  write_to_file(out_file, ne, nv, g.get_max_degree(), _vertices, _edges);
}

void build_med(Graph &g, vidType first, vidType last, vidType m_deg, vidType interval, std::string out_file) {
  vidType nv = last - first;
  vidType interval_key_len = (interval * 2) / 32;
  vector<vidType> edges_compressed;
  eidType *vertices_compressed = new eidType[nv+1];
  vertices_compressed[0] = 0;
  vidType *in_buffer = new vidType[m_deg];
  vector<vidType> out_buffer;
  vidType *out_ptr;
  vidType *key_ptr;
  vbyte_encoder vb_encoder("streamvbyte");

  for (vidType v = first; v < last; v++) {
    vidType relabel_v = v - first;
    vidType deg = g.decode_vertex_vbyte(v, in_buffer, "streamvbyte");
    edges_compressed.push_back(deg);
    if (out_buffer.size() < deg + 1024) out_buffer.resize(deg + 1024);
    uint32_t key_len8 = (deg + 3) / 4;
    uint32_t key_len32 = (key_len8 + 3) / 4;
    key_ptr = out_buffer.data();
    out_ptr = key_ptr + key_len32;
    vidType n_rounds = deg / interval;
    vidType prefix_size = 0;
    vidType total_prefix = 0;

    for (vidType r = 0; r < n_rounds; r++) {
      edges_compressed.push_back(total_prefix);
      prefix_size = vb_encoder.encode(interval, in_buffer, key_ptr, out_ptr);
      total_prefix += prefix_size;
      in_buffer += interval;
      key_ptr += interval_key_len;
      out_ptr += prefix_size;
    }

    vidType count = deg % interval;
    if (count != 0) {
      edges_compressed.push_back(total_prefix);
      prefix_size = vb_encoder.encode(count, in_buffer, key_ptr, out_ptr);
      // total_prefix += prefix_size;
      out_ptr += prefix_size;
    } 
    vidType total_size_v = out_ptr - out_buffer.data();
    edges_compressed.insert(edges_compressed.end(), out_buffer.begin(), out_buffer.begin() + total_size_v);
    vertices_compressed[relabel_v+1] = vertices_compressed[relabel_v] + 1 + ((deg + interval - 1) / interval) + total_size_v;
    in_buffer -= (deg - count);
  }
  vidType ne = edges_compressed.size();
  write_to_file(out_file, ne, nv, m_deg, vertices_compressed, &edges_compressed[0]);
}

void build_low(Graph &g, vidType first_v, eidType ne, vidType nv, vidType m_deg, std::string out_file) {
  eidType *_rowptr = new eidType[nv + 1];
  vidType *_colidx = new vidType[ne];
  eidType *base_rowptr = g._rowptr_compressed() + first_v;
  vidType *base_colidx = g._colidx_compressed() + base_rowptr[0];
  eidType diff = base_rowptr[0];
  for (vidType i = 1; i <= nv; i++) {
    _rowptr[i] = base_rowptr[i] - diff;
  }
  for (eidType e = 0; e < ne; e++) {
    _colidx[e] = base_colidx[e];
  }
  write_to_file(out_file, ne, nv, m_deg, _rowptr, _colidx);
}

void write_subgraphs(Graph &g, bool add_uncomp, size_t uncomp_mem, size_t total_mem, int low_deg, int high_deg, vidType prefix_interval, std::string out_prefix) {
  std::cout << "file name! " << out_prefix << std::endl;
  std::string file_specs = std::to_string(low_deg) + "_" + std::to_string(high_deg);
  // get high degree subgraph

  vidType last_uncomp = 0;
  vidType curr_deg = g.get_degree_vbyte(last_uncomp);
  eidType total_deg = curr_deg;
  while (total_deg < uncomp_mem) {
    last_uncomp++;
    curr_deg = g.get_degree_vbyte(last_uncomp);
    total_deg += curr_deg;
  }
  last_uncomp--;
  total_deg -= curr_deg;
  if (!add_uncomp) {
    total_deg = 0;
  }
  build_uncomp(g, last_uncomp + 1, total_deg, out_prefix + "u" + std::to_string(total_mem / uncomp_mem));
  std::cout << "Allocated uncompressed subgraph\n";

  // get high degree subgraph
  auto g_rptr = g.rowptr_compressed();
  size_t mem_left = (total_mem - uncomp_mem) * 3 / 4;
  vidType first_high = last_uncomp + 1;
  if (!add_uncomp) {
    first_high = 0;
  }
  vidType last_high = first_high;
  curr_deg = g.get_degree_vbyte(last_high);
  size_t curr_mem = g_rptr[first_high+1] - g_rptr[first_high];
  while (curr_mem < mem_left && curr_deg > high_deg) {
    last_high++;
    curr_deg = g.get_degree_vbyte(last_high);
    curr_mem += g_rptr[last_high+1] - g_rptr[last_high];
  }
  last_high--;
  std::cout << "high deg cutoff " << g.get_degree_vbyte(first_high) << " " << g.get_degree_vbyte(last_high) << " " << g.get_degree_vbyte(last_high+1) << std::endl;
  std::cout << "mem left " << mem_left << std::endl;
  build_med(g, first_high, last_high + 1, g.get_degree_vbyte(first_high), prefix_interval, out_prefix + "h" + std::to_string(high_deg));
  std::cout << "Allocated high subgraph\n";

  // get medium degree subgraph
  vidType first_med = last_high + 1;
  vidType last_med = first_med;
  while (g.get_degree_vbyte(last_med) > low_deg) {
    last_med++;
  }
  last_med--;
  build_med(g, first_med, last_med + 1, g.get_degree_vbyte(first_med), prefix_interval, out_prefix + "m" + file_specs);
  std::cout << "Allocated medium subgraph\n";

  // get low degree subgraph
  vidType first_low = last_med + 1;
  // auto g_rptr = g.rowptr_compressed();
  total_deg = g_rptr[g.V()] - g_rptr[first_low];
  size_t mem_vert = size_t(g.V() - first_low + 1)*sizeof(eidType);
  size_t mem_edge = size_t(total_deg)*sizeof(vidType);
  size_t mem_graph = mem_vert + mem_edge;
  std::cout << "Low deg subgraph: " << (float)mem_graph / (float)1000000000 << "GB; |V| " << g.V() - first_low << "\n";

  build_low(g, first_low, total_deg, g.V() - first_low, low_deg, out_prefix + "l" + std::to_string(low_deg));
  std::cout << "Allocated low subgraph\n";
  std::cout << "First low is " << first_low << " first med is " << first_med << std::endl;
}

int main(int argc, char* argv[]) {
  // size_t memsize = print_device_info(0);
  std::string in_prefix = argv[1];
  std::string out_prefix = argv[2];
  // std::string out_prefix = argv[2];
  std::string scheme = "streamvbyte";
  bool permutated = false;
  int c;
  int pdeg = BLOCK_SIZE;
  int low_deg = 32;
  int high_deg = 256;
  vidType prefix_interval = WARP_SIZE;
  size_t gpu_mem = 10000000000;
  int uncomp_mem_ratio = 4;
  size_t gpu_mem_uncomp = gpu_mem / uncomp_mem_ratio;
  bool add_uncomp = true;
  while ((c = getopt(argc, argv, "d:l:h:v:a")) != -1) {
    switch (c) {
      case 'd':
        pdeg = atoi(optarg);
        break;
      case 'l':
        low_deg = atoi(optarg);
        break;
      case 'h':
        high_deg = atoi(optarg);
        break;
      case 'v':
        prefix_interval = (vidType)atoi(optarg);
        break;
      case 'a':
        add_uncomp = false;
        break;
      default:
        abort();
    }
  }
  std::cout << "LOADED COMPRESSED GRAPH\n" << std::endl;
  Graph g;
  g.load_compressed_graph(in_prefix, scheme, permutated);
  write_subgraphs(g, add_uncomp, gpu_mem_uncomp, gpu_mem, low_deg, high_deg, prefix_interval, out_prefix);
  return 0;
}
