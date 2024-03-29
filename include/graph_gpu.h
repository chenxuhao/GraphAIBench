#pragma once
#include "graph.h"
#include "scan.h"
#include "operations.cuh"
#include "cutil_subset.h"
#include "cuda_profiler_api.h"

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

class GraphGPU {
protected:
  bool is_directed_;                // is it a directed graph?
  bool has_reverse;                 // has reverse/incoming edges maintained
  vidType num_vertices;             // number of vertices
  eidType num_edges;                // number of edges
  int device_id, n_gpu;             // no. of GPUs
  int num_vertex_classes;           // number of unique vertex labels
  int num_edge_classes;             // number of unique edge labels
  vidType max_degree;               // maximun degree
  eidType *d_rowptr, *d_in_rowptr;  // row pointers of CSR format
  vidType *d_colidx, *d_in_colidx;  // column induces of CSR format
  vidType *d_src_list, *d_dst_list; // for COO format
  vlabel_t *d_vlabels;              // vertex labels
  elabel_t *d_elabels;              // edge labels
  vidType *d_degrees;               // vertex degree
  vidType *d_vlabels_frequency;     // vertex label frequency
  vidType *d_adj_buffer;            // buffer for copying an adjacency list from a remote GPU
public:
  GraphGPU(vidType nv, eidType ne, int vl=0, int el=0, 
           int n=0, int m=1, bool use_nvshmem=false) :
      GraphGPU(n, m, nv, ne, vl, el) {
    if (nv>0 && ne>0 && !use_nvshmem) allocateFrom(nv, ne, vl, el);
  }
  GraphGPU(Graph &g, int n=0, int m=1) : 
      GraphGPU(n, m, g.V(), g.E(), g.get_vertex_classes(), g.get_edge_classes()) {
    init(g);
  }
  GraphGPU(int n=0, int m=0, vidType nv=0, eidType ne=0, int vl=1, int el=1,
           bool directed=false, bool reverse=false) : 
      is_directed_(directed),
      has_reverse(reverse),
      num_vertices(nv),
      num_edges(ne),
      device_id(n), n_gpu(m),
      num_vertex_classes(vl),
      num_edge_classes(el),
      max_degree(0),
      d_rowptr(NULL),
      d_in_rowptr(NULL),
      d_colidx(NULL), 
      d_in_colidx(NULL), 
      d_src_list(NULL),
      d_dst_list(NULL),
      d_vlabels(NULL),
      d_elabels(NULL), 
      d_degrees(NULL), 
      d_vlabels_frequency(NULL),
      d_adj_buffer(NULL) {
  }
  void release() { clean(); clean_edgelist(); clean_labels(); }
  inline __device__ __host__ bool is_directed() { return is_directed_; }
  inline __device__ __host__ int get_num_devices() { return n_gpu; }
  inline __device__ __host__ vidType* get_buffer_ptr() { return d_adj_buffer; }
  inline __device__ __host__ vidType V() { return num_vertices; }
  inline __device__ __host__ vidType size() { return num_vertices; }
  inline __device__ __host__ eidType E() { return num_edges; }
  inline __device__ __host__ eidType sizeEdges() { return num_edges; }
  inline __device__ __host__ vidType get_max_degree() { return max_degree; }
  inline __device__ __host__ bool valid_vertex(vidType vertex) { return (vertex < num_vertices); }
  inline __device__ __host__ bool valid_edge(eidType edge) { return (edge < num_edges); }
  inline __device__ __host__ vidType get_src(eidType eid) const { return d_src_list[eid]; }
  inline __device__ __host__ vidType get_dst(eidType eid) const { return d_dst_list[eid]; }
  inline __device__ __host__ vidType* get_src_ptr(eidType eid) const { return d_src_list; }
  inline __device__ __host__ vidType* get_dst_ptr(eidType eid) const { return d_dst_list; }
  inline __device__ __host__ vidType* N(vidType vid) { return d_colidx + d_rowptr[vid]; }
  inline __device__ __host__ vidType N(vidType v, eidType e) { return d_colidx[d_rowptr[v] + e]; }
  inline __device__ __host__ eidType* rowptr() { return d_rowptr; }
  inline __device__ __host__ vidType* colidx() { return d_colidx; }
  inline __device__ __host__ eidType* out_rowptr() { return d_rowptr; }
  inline __device__ __host__ vidType* out_colidx() { return d_colidx; }
  inline __device__ __host__ eidType* in_rowptr() { return d_in_rowptr; }
  inline __device__ __host__ vidType* in_colidx() { return d_in_colidx; }
  inline __device__ __host__ eidType getOutDegree(vidType src) { return d_rowptr[src+1] - d_rowptr[src]; }
  inline __device__ __host__ eidType getInDegree(vidType src) { return d_in_rowptr[src+1] - d_in_rowptr[src]; }
  inline __device__ __host__ vidType get_degree(vidType src) { return vidType(d_rowptr[src+1] - d_rowptr[src]); }
  inline __device__ __host__ vidType getEdgeDst(eidType edge) { return d_colidx[edge]; }
  inline __device__ __host__ vidType getOutEdgeDst(eidType edge) { return d_colidx[edge]; }
  inline __device__ __host__ vidType getInEdgeDst(eidType edge) { return d_in_colidx[edge]; }
  inline __device__ __host__ eidType edge_begin(vidType src) { return d_rowptr[src]; }
  inline __device__ __host__ eidType edge_end(vidType src) { return d_rowptr[src+1]; }
  inline __device__ __host__ eidType out_edge_begin(vidType src) { return d_rowptr[src]; }
  inline __device__ __host__ eidType out_edge_end(vidType src) { return d_rowptr[src+1]; }
  inline __device__ __host__ eidType in_edge_begin(vidType src) { return d_in_rowptr[src]; }
  inline __device__ __host__ eidType in_edge_end(vidType src) { return d_in_rowptr[src+1]; }
  inline __device__ __host__ vlabel_t getData(vidType vid) { return d_vlabels[vid]; }
  inline __device__ __host__ elabel_t getEdgeData(eidType eid) { return d_elabels[eid]; }
  inline __device__ __host__ vidType getLabelsFrequency(vlabel_t label) { return d_vlabels_frequency[label]; }
  inline __device__ __host__ vlabel_t* getVlabelPtr() { return d_vlabels; }
  inline __device__ __host__ elabel_t* getElabelPtr() { return d_elabels; }
  inline __device__ __host__ vlabel_t* get_vlabel_ptr() { return d_vlabels; }
  inline __device__ __host__ elabel_t* get_elabel_ptr() { return d_elabels; }
 
  inline __device__ __host__ bool is_freq_vertex(vidType v, int threshold) {
    auto label = int(d_vlabels[v]);
    assert(label <= num_vertex_classes);
    if (d_vlabels_frequency[label] >= threshold) return true;
    return false;
  }
  void clean() {
    if (d_rowptr != NULL)
      CUDA_SAFE_CALL(cudaFree(d_rowptr));
    if (d_colidx != NULL)
      CUDA_SAFE_CALL(cudaFree(d_colidx));
  }
  void clean_edgelist() {
    if (d_src_list != NULL)
      CUDA_SAFE_CALL(cudaFree(d_src_list));
    if (d_dst_list != NULL)
      CUDA_SAFE_CALL(cudaFree(d_dst_list));
  }
  void clean_labels() {
    if (d_vlabels != NULL)
      CUDA_SAFE_CALL(cudaFree(d_vlabels));
    if (d_elabels != NULL)
      CUDA_SAFE_CALL(cudaFree(d_elabels));
    if (d_vlabels_frequency != NULL)
      CUDA_SAFE_CALL(cudaFree(d_vlabels_frequency));
  }
  void allocateFrom(vidType nv, eidType ne, bool has_vlabel = false, 
                    bool has_elabel = false, bool use_uva = false, bool has_reverse = false) {
    std::cout << "Allocating GPU memory for the graph ... ";
    if (use_uva) {
      CUDA_SAFE_CALL(cudaMallocManaged(&d_rowptr, (nv+1) * sizeof(eidType)));
      CUDA_SAFE_CALL(cudaMallocManaged(&d_colidx, ne * sizeof(vidType)));
      if (has_reverse) {
        CUDA_SAFE_CALL(cudaMallocManaged(&d_in_rowptr, (nv+1) * sizeof(eidType)));
        CUDA_SAFE_CALL(cudaMallocManaged(&d_in_colidx, ne * sizeof(vidType)));
      }
    } else {
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr, (nv+1) * sizeof(eidType)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx, ne * sizeof(vidType)));
      if (has_reverse) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_rowptr, (nv+1) * sizeof(eidType)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_colidx, ne * sizeof(vidType)));
      }
    }
    if (has_vlabel)
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_vlabels, nv * sizeof(vlabel_t)));
    if (has_elabel)
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_elabels, ne * sizeof(elabel_t)));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cout << "Done\n";
  }
#ifdef USE_NVSHMEM
  void allocate_nvshmem(vidType nv, eidType ne, vidType max_degree, int id,
                        bool has_vlabel = false, 
                        bool has_elabel = false, 
                        bool has_reverse = false) {
    //std::cout << "Allocating NVSHMEM symmetric memory for the graph\n";
    std::cout << "Allocating NVSHMEM memory for rowptr: " << nv+1 << " \n";
    d_rowptr = (eidType*)nvshmem_malloc((nv+1) * sizeof(eidType));
    std::cout << "Allocated rowptr pointer " << d_rowptr << "\n";
    std::cout << "Allocating NVSHMEM memory for colidx: " << ne << "\n";
    d_colidx = (vidType*)nvshmem_malloc(ne * sizeof(vidType));
    std::cout << "Allocated colidx pointer " << d_colidx << "\n";
    //std::cout << "Allocating NVSHMEM memory for buffer\n";
    //d_adj_buffer = (vidType*)nvshmem_malloc(max_degree * sizeof(vidType));
    //std::cout << "Zerolizing NVSHMEM memory buffer\n";
    //cudaMemset(d_adj_buffer, 0, max_degree * sizeof(vidType));
  }
#endif
  void copyToDevice(vidType nv, eidType ne, eidType *h_rowptr, vidType *h_colidx, bool reverse = false,
                    label_t* h_vlabels = NULL, elabel_t* h_elabels = NULL, bool use_uva = false) {
    std::cout << "Copying graph data to GPU memory ... ";
    auto rptr = d_rowptr;
    auto cptr = d_colidx;
    if (reverse) {
      rptr = d_in_rowptr;
      cptr = d_in_colidx;
    }
    if (use_uva) {
      if (h_rowptr != NULL)
        std::copy(h_rowptr, h_rowptr+nv+1, rptr);
      if (h_colidx != NULL)
        std::copy(h_colidx, h_colidx+ne, cptr);
    } else {
      if (h_rowptr != NULL)
        CUDA_SAFE_CALL(cudaMemcpy(rptr, h_rowptr, (nv+1) * sizeof(eidType), cudaMemcpyHostToDevice));
      if (h_colidx != NULL)
        CUDA_SAFE_CALL(cudaMemcpy(cptr, h_colidx, ne * sizeof(vidType), cudaMemcpyHostToDevice));
      if (h_vlabels != NULL)
        CUDA_SAFE_CALL(cudaMemcpy(d_vlabels, h_vlabels, nv * sizeof(vlabel_t), cudaMemcpyHostToDevice));
      if (h_elabels != NULL)
        CUDA_SAFE_CALL(cudaMemcpy(d_elabels, h_elabels, ne * sizeof(elabel_t), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    std::cout << "Done\n";
  }
  void init(Graph &g, int n, int m) {
    device_id = n;
    n_gpu = m;
    init(g);
  }
  void init(Graph &hg) {
    auto nv = hg.num_vertices();
    auto ne = hg.num_edges();
    size_t mem_vert = size_t(nv+1)*sizeof(eidType);
    size_t mem_edge = size_t(ne)*sizeof(vidType);
    size_t mem_graph = mem_vert + mem_edge;
    size_t mem_el = mem_edge; // memory for the edgelist
    size_t mem_all = mem_graph + mem_el;
    auto mem_gpu = get_gpu_mem_size();
    bool use_uva = mem_all > mem_gpu;
    auto v_classes = hg.get_vertex_classes();
    auto h_vlabel_freq = hg.get_label_freq_ptr();
    max_degree = hg.get_max_degree();
    Timer t;
    t.Start();
    allocateFrom(nv, ne, hg.has_vlabel(), hg.has_elabel(), use_uva, hg.has_reverse_graph());
    t.Stop();
    std::cout << "Time on allocating device memory for the graph on GPU" << device_id << ": " << t.Seconds() << " sec\n";
    t.Start();
    copyToDevice(nv, ne, hg.out_rowptr(), hg.out_colidx(), false, hg.getVlabelPtr(), hg.getElabelPtr(), use_uva);
    if (hg.has_vlabel()) {
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_vlabels_frequency, (v_classes+1) * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMemcpy(d_vlabels_frequency, h_vlabel_freq, (v_classes+1) * sizeof(vidType), cudaMemcpyHostToDevice));
    }
    if (hg.has_reverse_graph()) {
      has_reverse = true;
      if (hg.is_directed()) {
        std::cout << "This graph maintains both incomming and outgoing edge-list\n";
        copyToDevice(nv, ne, hg.in_rowptr(), hg.in_colidx(), true);
      } else { // undirected graph
        d_in_rowptr = d_rowptr;
        d_in_colidx = d_colidx;
      }
    }
    t.Stop();
    std::cout << "Time on copying graph to GPU" << device_id << ": " << t.Seconds() << " sec\n";
  }
  void init_nvshmem(const Graph &hg, int id) {
    auto nv = hg.V();
    auto ne = hg.E();
    std::cout << "Copying subgraph[" << id << "]: nv = " << nv << " ne = " << ne << "\n";
    assert(hg.rowptr());
    assert(hg.colidx());
    Timer t;
    t.Start();
    //CUDA_SAFE_CALL(cudaSetDevice(id));
    std::cout << "Copying subgraph[" << id << "]'s rowptr\n";
    std::cout << "d_rowptr pointer " << d_rowptr << "\n";
    CUDA_SAFE_CALL(cudaMemcpy(d_rowptr, hg.rowptr(), (nv+1) * sizeof(eidType), cudaMemcpyHostToDevice));
    std::cout << "Copying subgraph[" << id << "]'s colidx\n";
    std::cout << "d_colidx pointer " << d_colidx << "\n";
    CUDA_SAFE_CALL(cudaMemcpy(d_colidx, hg.colidx(), ne * sizeof(vidType), cudaMemcpyHostToDevice));
    t.Stop();
    std::cout << "Time on copying the subgraph to GPU_" << id << ": " << t.Seconds() << " sec\n";
  }
  void toHost(Graph &hg) {
    auto nv = num_vertices;
    auto ne = num_edges;
    hg.allocateFrom(nv, ne);
    auto h_rowptr = hg.out_rowptr();
    auto h_colidx = hg.out_colidx();
    CUDA_SAFE_CALL(cudaMemcpy(h_rowptr, d_rowptr, (nv+1) * sizeof(eidType), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_colidx, d_colidx, ne * sizeof(vidType), cudaMemcpyDeviceToHost));
  }
  // this is for single-GPU only
  size_t init_edgelist(Graph &hg, bool sym_break = false, bool ascend = false, bool zero_copy = false) {
    auto nnz = sym_break ? num_edges/2 : num_edges;
    size_t mem_el = size_t(nnz)*sizeof(vidType);
    auto mem_gpu = get_gpu_mem_size();
    if (zero_copy || mem_el > mem_gpu) {
      //init_edgelist_um(hg, sym_break);
      init_edgelist_zero_copy(hg, sym_break);
    } else {
      hg.init_edgelist(sym_break, ascend);
      copy_edgelist_to_device(nnz, hg.get_src_ptr(), hg.get_dst_ptr(), sym_break);
    }
    return nnz;
  }
  void copy_edgelist_to_device(size_t nnz, Graph &hg, bool sym_break = false) {
    copy_edgelist_to_device(0, nnz, hg, sym_break);
  }
  void copy_edgelist_to_device(size_t begin, size_t end, Graph &hg, bool sym_break = false) {
    copy_edgelist_to_device(begin, end, hg.get_src_ptr(), hg.get_dst_ptr(), sym_break);
  }
  void copy_edgelist_to_device(size_t nnz, vidType* h_src_list, vidType* h_dst_list, bool sym_break) {
    copy_edgelist_to_device(0, nnz, h_src_list, h_dst_list, sym_break);
  }
  void copy_edgelist_to_device(size_t begin, size_t end, vidType* h_src_list, vidType* h_dst_list, bool sym_break) {
    auto n = end - begin;
    eidType n_tasks_per_gpu = eidType(n-1) / eidType(n_gpu) + 1;
    eidType start = begin + device_id * n_tasks_per_gpu;
    if (!sym_break) d_dst_list = d_colidx + start;
    eidType num = n_tasks_per_gpu;
    if (start + num > end) num = end - start;
    //std::cout << "Allocating edgelist on GPU" << device_id << " size = " << num 
    //          << " [" << start << ", " << start+num << ")\n";
    //Timer t;
    //t.Start();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_src_list, num * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_src_list, h_src_list+start, num * sizeof(vidType), cudaMemcpyHostToDevice));
    if (sym_break) {
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_dst_list, num * sizeof(vidType)));
      CUDA_SAFE_CALL(cudaMemcpy(d_dst_list, h_dst_list+start, num * sizeof(vidType), cudaMemcpyHostToDevice));
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //t.Stop();
    //std::cout << "Time on copying edgelist to GPU" << device_id << ": " << t.Seconds() << " sec\n";
  }
  void copy_edgelist_to_device(std::vector<eidType> lens, std::vector<vidType*> &srcs, std::vector<vidType*> &dsts) {
    //Timer t;
    //t.Start();
    vidType* src_ptr = srcs[device_id];
    vidType* dst_ptr = dsts[device_id];
    auto num = lens[device_id];
    //std::cout << "src_ptr = " << src_ptr << " dst_ptr = " << dst_ptr << "\n";
    //std::cout << "Allocating edgelist on GPU" << device_id << " size = " << num << "\n";
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_src_list, num * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_src_list, src_ptr, num * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_dst_list, num * sizeof(vidType)));
    CUDA_SAFE_CALL(cudaMemcpy(d_dst_list, dst_ptr, num * sizeof(vidType), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //t.Stop();
    //std::cout << "Time on copying edgelist to GPU" << device_id << ": " << t.Seconds() << " sec\n";
  }
  void init_edgelist_um(Graph &g, bool sym_break = false) {
    Timer t;
    t.Start();
    auto nnz = sym_break ? g.E()/2 : g.E();
    CUDA_SAFE_CALL(cudaMallocManaged(&d_src_list, nnz * sizeof(vidType)));
    if (!sym_break) d_dst_list = d_colidx;
    else CUDA_SAFE_CALL(cudaMallocManaged(&d_dst_list, nnz * sizeof(vidType)));
    size_t i = 0;
    for (vidType v = 0; v < g.V(); v ++) {
      for (auto u : g.N(v)) {
        assert(u != v);
        if (sym_break && v < u) break;  
        d_src_list[i] = v;
        if (sym_break) d_dst_list[i] = u;
        i ++;
      }
    }
    t.Stop();
    std::cout << "Time generating the edgelist on CUDA unified memory: " << t.Seconds() << " sec\n";
  }
  void init_edgelist_zero_copy(Graph &g, bool sym_break = false) {
    std::cout << "Initializing edgelist using zero-copy\n";
    Timer t;
    t.Start();
    size_t NUM_BYTES = (sym_break ? g.E()/2 : g.E()) * sizeof(vidType);
    vidType *h_src_list, *h_dst_list;
    cudaHostAlloc((void**)&h_src_list, NUM_BYTES, cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_dst_list, NUM_BYTES, cudaHostAllocMapped);
    if (sym_break) {
      VertexList degrees(g.V(), 0);
      #pragma omp parallel for
      for (vidType v = 0; v < g.V(); v ++) {
        for (auto u : g.N(v)) {
          if (v < u) break;
          degrees[v] ++;
        }
      }
      eidType *offsets = custom_alloc_global<eidType>(g.V()+1);
      parallel_prefix_sum<vidType,eidType>(degrees, offsets);
      #pragma omp parallel for
      for (vidType v = 0; v < g.V(); v ++) {
        auto idx = offsets[v];
        for (auto u : g.N(v)) {
          if (v < u) break;
          h_src_list[idx] = v;
          h_dst_list[idx] = u;
          idx ++;
        }
      }
    } else {
      std::cout << "Initializing edgelist\n";
      #pragma omp parallel for
      for (vidType v = 0; v < g.V(); v ++) {
        auto idx = g.edge_begin(v);
        for (auto u : g.N(v)) {
          h_src_list[idx] = v;
          h_dst_list[idx] = u;
          idx ++;
        }
      }
    }
    cudaHostGetDevicePointer((void**)&d_src_list, (void*)h_src_list, 0);
    cudaHostGetDevicePointer((void**)&d_dst_list, (void*)h_dst_list, 0);
    t.Stop();
    std::cout << "Time generating the edgelist on CUDA unified memory: " << t.Seconds() << " sec\n";
  }

  // using a warp to compute the intersection of the neighbor lists of two vertices
  inline __device__ vidType warp_intersect(vidType v, vidType u) {
    vidType count = 0;
    assert(v != u);
    vidType v_degree = getOutDegree(v);
    vidType u_degree = getOutDegree(u);
    if (v_degree == 0 || u_degree == 0) return 0;
    vidType* adj_v = d_colidx + edge_begin(v);
    vidType* adj_u = d_colidx + edge_begin(u);
    count = intersect_num(adj_v, v_degree, adj_u, u_degree);
    return count;
  }

  // using a CTA to compute the intersection of the neighbor lists of two vertices
  inline __device__ vidType cta_intersect(vidType v, vidType u) {
    vidType count = 0;
    assert(v != u);
    vidType v_degree = getOutDegree(v);
    vidType u_degree = getOutDegree(u);
    if (v_degree == 0 || u_degree == 0) return 0;
    vidType* adj_v = d_colidx + edge_begin(v);
    vidType* adj_u = d_colidx + edge_begin(u);
    count = intersect_num_cta(adj_v, v_degree, adj_u, u_degree);
    return count;
  }
};

