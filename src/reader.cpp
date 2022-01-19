#include "util.h"
#include "reader.h"
#include "configs.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>  /* For O_RDWR */
#include <unistd.h> /* For open(), creat() */
#include <fstream>
#include <cassert>

// labels contain the ground truth (e.g. vertex classes) for each example
// (num_examples x 1). Note that labels is not one-hot encoded vector and it can
// be computed as y.argmax(axis=1) from one-hot encoded vector (y) of labels if
// required.
size_t Reader::read_labels(std::vector<label_t>& labels, bool is_single_class) {
  Timer t_read;
  t_read.Start();
  std::string filename = path + dataset_str + "-labels.txt";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  size_t m, num_classes; // m: number of samples
  in >> m >> num_classes >> std::ws;
  if (!is_single_class) {
    std::cout << "Using multi-class (multi-hot) labels\n";
    labels.resize(m * num_classes); // multi-class label for each vertex: N x E
  } else {
    std::cout << "Using single-class (one-hot) labels\n";
    labels.resize(m); // single-class (one-hot) label for each vertex: N x 1
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
          labels[v] = idx;
          break;
        }
      } else {
        labels[v * num_classes + idx] = x;
      }
    }
    v++;
  }
  in.close();
  t_read.Stop();
  // print the number of vertex classes
  std::cout << ", time: " << t_read.Millisecs() << " ms\n";
  return num_classes;
}

//! Read features, return the length of a feature vector
//! Features are stored in the Context class
size_t Reader::read_features(std::vector<float>& feats, std::string filetype) {
  std::cout << "Reading features ... ";
  Timer t_read;
  t_read.Start();
  size_t m, feat_len; // m = number of vertices
  std::string filename = path + dataset_str + ".ft";
  std::ifstream in;

  if (filetype == "bin") {
    std::string file_dims = path + dataset_str + "-dims.txt";
    std::ifstream ifs;
    ifs.open(file_dims, std::ios::in);
    ifs >> m >> feat_len >> std::ws;
    ifs.close();
  } else {
    in.open(filename, std::ios::in);
    in >> m >> feat_len >> std::ws;
  }
  std::cout << "N x D: " << m << " x " << feat_len << "\n";
  feats.resize(m*feat_len);
  if (filetype == "bin") {
    filename = path + dataset_str + "-feats.bin";
    in.open(filename, std::ios::binary | std::ios::in);
    in.read((char*)&feats[0], sizeof(float) * m * feat_len);
  } else {
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream edge_stream(line);
      size_t u, v;
      float w;
      edge_stream >> u;
      edge_stream >> v;
      edge_stream >> w;
      feats[u * feat_len + v] = w;
    }
  }
  in.close();
  t_read.Stop();
  std::cout << "Done, feature length: " << feat_len
            << ", time: " << t_read.Millisecs() << " ms\n";
  return feat_len;
}

//! Get masks from datafile where first line tells range of
//! set to create mask from
size_t Reader::read_masks(std::string mask_type, size_t n, size_t& begin,
                          size_t& end, mask_t* masks) {
  bool dataset_found = false;
  for (int i = 0; i < NUM_DATASETS; i++) {
    if (dataset_str == dataset_names[i]) {
      dataset_found = true;
      break;
    }
  }
  if (!dataset_found) {
    std::cout << "Dataset currently not supported\n";
    exit(1);
  }
  size_t i             = 0;
  size_t sample_count  = 0;
  std::string filename = path + dataset_str + "-" + mask_type + "_mask.txt";
  // std::cout << "Reading " << filename << "\n";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  in >> begin >> end >> std::ws;
  while (std::getline(in, line)) {
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
  std::cout << mask_type << "_mask range: [" << begin << ", "
            << end << ") Number of valid samples: " << sample_count << " ("
            << (float)sample_count / (float)n * (float)100 << "\%)\n";
  in.close();
  return sample_count;
}

void Reader::progressPrint(unsigned max, unsigned i) {
  const unsigned nsteps = 10;
  unsigned ineachstep   = (max / nsteps);
  if (ineachstep == 0)
    ineachstep = 1;
  if (i % ineachstep == 0) {
    int progress = ((size_t)i * 100) / max + 1;
    printf("\t%3d%%\r", progress);
    fflush(stdout);
  }
}

void Reader::readGraphFromGRFile(LearningGraph* g) {
  std::cout << "Reading graph into CPU memory\n";
  std::string filename = path + dataset_str + ".csgr";
  std::ifstream ifs;
  ifs.open(filename);
  int masterFD = open(filename.c_str(), O_RDONLY);
  if (masterFD == -1) {
    std::cout << "LearningGraph: unable to open " << filename << "\n";
    exit(1);
  }
  struct stat buf;
  int f = fstat(masterFD, &buf);
  if (f == -1) {
    std::cout << "LearningGraph: unable to stat " << filename << "\n";
    exit(1);
  }
  size_t masterLength = buf.st_size;
  int _MAP_BASE       = MAP_PRIVATE;
  void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (m == MAP_FAILED) {
    m = 0;
    std::cout << "LearningGraph: mmap failed.\n";
    exit(1);
  }
  Timer t;
  t.Start();

  uint64_t* fptr                           = (uint64_t*)m;
  __attribute__((unused)) uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  uint64_t nv         = le64toh(*fptr++);
  uint64_t ne         = le64toh(*fptr++);
  uint64_t* outIdx    = fptr;
  fptr += nv;
  uint32_t* fptr32 = (uint32_t*)fptr;
  uint32_t* outs   = fptr32;
  fptr32 += ne;
  if (ne % 2)
    fptr32 += 1;
  if (sizeEdgeTy != 0) {
    std::cout << "LearningGraph: currently edge data not supported.\n";
    exit(1);
  }
  g->allocateFrom(nv, ne);
  auto rowptr = g->row_start_host_ptr();
  for (unsigned vid = 0; vid < nv; ++vid) {
    g->fixEndEdge(vid, le64toh(outIdx[vid]));
    auto degree = rowptr[vid + 1] - rowptr[vid];
    for (unsigned jj = 0; jj < degree; ++jj) {
      unsigned eid = rowptr[vid] + jj;
      unsigned dst = le32toh(outs[eid]);
      if (dst >= nv) {
        printf("\tinvalid edge from %d to %d at index %d(%d).\n", vid, dst, jj,
               eid);
        exit(0);
      }
      g->constructEdge(eid, dst);
    }
    progressPrint(nv, vid);
  }
  ifs.close();
  t.Stop();
  auto runtime = t.Millisecs();
  g->degree_counting();
  std::cout << "read " << masterLength << " bytes in " << runtime << " ms ("
            << masterLength / 1000.0 / runtime << " MB/s)\n"
            << "|V| " << nv << " |E| " << ne << " max_deg " << g->get_max_degree() << "\n";
}

