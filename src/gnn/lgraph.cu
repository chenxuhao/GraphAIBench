#include "lgraph.h"
#include "cutils.h"
#include "math_functions.hh"

// computing normalization factor for each vertex
__global__ void compute_vertex_data_kernel(int n, GraphGPU graph) {
  CUDA_KERNEL_LOOP(i, n) {
    float temp = sqrt(float(graph.getDegree(i)));
    if (temp == 0.0) graph.set_vertex_data(i, 0.0);
    else graph.set_vertex_data(i, 1.0 / temp);
  }
}

// computing normalization factor for each edge
__global__ void compute_gcn_edge_data_kernel(int n, GraphGPU graph) {
  CUDA_KERNEL_LOOP(src, n) {
    assert(src < n);
    float c_i = sqrt(float(graph.getDegree(src)));
    auto start  = graph.edge_begin(src);
    index_t end = graph.edge_end(src);
    for (index_t e = start; e != end; e++) {
      index_t dst = graph.getEdgeDst(e);
      assert(dst < n);
      float c_j = sqrt(float(graph.getDegree(dst)));
      float c_ij = 0.;
      if (c_j != 0) c_ij = 1.0 / (c_i * c_j);
      graph.set_edge_data(e, c_ij);
    }
  }
}

// computing normalization factor for each edge
__global__ void compute_mean_edge_data_kernel(int n, GraphGPU graph, bool trans) {
  CUDA_KERNEL_LOOP(src, n) {
    assert(src < n);
    float c = 0.;
    if (!trans) c = 1. / float(graph.getDegree(src));
    auto start  = graph.edge_begin(src);
    index_t end = graph.edge_end(src);
    for (index_t e = start; e != end; e++) {
      index_t dst = graph.getEdgeDst(e);
      assert(dst < n);
      if (trans) {
        c = 1. / float(graph.getDegree(dst));
        graph.set_trans_edge_data(e, c);
      } else graph.set_edge_data(e, c);
    }
  }
}

void LearningGraph::alloc_on_device() {
  //std::cout << "Allocating graph memory on GPU\n";
  if (d_rowptr_ && gpu_vsize < num_vertices_) uint_free_device(d_rowptr_);
  if (d_rowptr_ == NULL) {
    uint_malloc_device(num_vertices_+1, d_rowptr_);
    CudaTest("Allocating row pointers memory on GPU failed");
    gpu_vsize = num_vertices_;
    //std::cout << "row pointers addr: 0x" << std::hex << d_rowptr_ << std::dec << "\n";
  }
  //std::cout << "before: ne=" << num_edges_ << ", gpu_esize=" << gpu_esize << "\n";
  //std::cout << "column indices addr: 0x" << std::hex << d_colidx_ << std::dec << "\n";
  if (d_colidx_ && gpu_esize < num_edges_) uint_free_device(d_colidx_);
  if (d_colidx_ == NULL) {
    //std::cout << "Allocating column indices memory on GPU\n";
    uint_malloc_device(num_edges_, d_colidx_);
    CudaTest("Allocating column indices memory on GPU failed");
    gpu_esize = num_edges_;
    //std::cout << "column indices addr: 0x" << std::hex << d_colidx_ << std::dec << "\n";
  }
  //std::cout << "after: ne=" << num_edges_ << ", gpu_esize=" << gpu_esize << "\n";
  //std::cout << "column indices addr: 0x" << std::hex << d_colidx_ << std::dec << "\n";
}

void LearningGraph::alloc_on_device(index_t n) {
  uint_malloc_device(n+1, d_rowptr_);
  gpu_vsize = n;
  CudaTest("Allocating row pointers memory on GPU failed");
}

void LearningGraph::copy_to_gpu() {
  assert(d_rowptr_);
  assert(d_colidx_);
  //std::cout << "Copying graph row pointers (nv=" << num_vertices_ << " , gpu_vsize=" << gpu_vsize << ") from CPU memory to GPU memory\n";
  //copy_uint_device(num_vertices_+1, row_start_host_ptr(), d_rowptr_);
  copy_async_device<index_t>(num_vertices_+1, row_start_host_ptr(), d_rowptr_);
  //std::cout << "Copying graph column indices (ne=" << num_edges_ << " , gpu_esize=" << gpu_esize << ") from CPU memory to GPU memory\n";
  //copy_uint_device(num_edges_, edge_dst_host_ptr(), d_colidx_);
  copy_async_device<index_t>(num_edges_, edge_dst_host_ptr(), d_colidx_);
  //if (!vertex_data_.empty()) copy_float_device(num_vertices_, &vertex_data_[0], d_vertex_data_);
  //if (!edge_data_.empty()) copy_float_device(num_edges_, &edge_data_[0], d_edge_data_);
  //print_test();
} 

void LearningGraph::compute_vertex_data() {
  auto n = num_vertices_;
  //std::cout << "Pre-computing normalization constants (nv=" << n << ") ... ";
  if (d_vertex_data_ && vdata_size < n) float_free_device(d_vertex_data_); // graph size may change due to subgraph sampling
  if (d_vertex_data_ == NULL) {
    float_malloc_device(n, d_vertex_data_);
    vdata_size = n;
  }
  init_const_gpu(n, 0.0, d_vertex_data_);
  compute_vertex_data_kernel<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>(n, *this);
  CudaTest("solving compute_vertex_data kernel failed");
}

void LearningGraph::compute_edge_data() {
  auto ne = num_edges_;
  auto nv = num_vertices_;
  //std::cout << "Pre-computing edge normalization constants (nv=" << nv << ", ne=" << ne << ") ... ";
  if (d_edge_data_ && edata_size < ne)
    float_free_device(d_edge_data_); // graph size may change due to subgraph sampling
#ifdef USE_SAGE
  if (d_trans_edge_data_ && edata_size < ne)
    float_free_device(d_trans_edge_data_); // graph size may change due to subgraph sampling
#endif
  if (d_edge_data_ == NULL) {
    float_malloc_device(ne, d_edge_data_);
    edata_size = ne;
  }
#ifdef USE_SAGE
  if (d_trans_edge_data_ == NULL) {
    float_malloc_device(ne, d_trans_edge_data_);
    edata_size = ne;
  }
#endif
  init_const_gpu(ne, 0.0, d_edge_data_);
#ifdef USE_SAGE
  init_const_gpu(ne, 0.0, d_trans_edge_data_);
  CudaTest("Initializing SAGE edge_data failed");
  compute_mean_edge_data_kernel<<<CUDA_GET_BLOCKS(nv), CUDA_NUM_THREADS>>>(nv, *this, false);
  CudaTest("solving SAGE compute_edge_data kernel failed");
  compute_mean_edge_data_kernel<<<CUDA_GET_BLOCKS(nv), CUDA_NUM_THREADS>>>(nv, *this, true);
  CudaTest("solving SAGE transposed compute_edge_data kernel failed");
#else
  CudaTest("Initializing edge_data failed");
  compute_gcn_edge_data_kernel<<<CUDA_GET_BLOCKS(nv), CUDA_NUM_THREADS>>>(nv, *this);
  CudaTest("solving compute_edge_data kernel failed");
#endif
}

void LearningGraph::print_test() {
  printf("d_rowptr_: 0x%x\n", d_rowptr_);
  printf("d_colidx_: 0x%x\n", d_colidx_);
  auto nv = num_vertices_;
  auto ne = num_edges_;
  if (nv > 10) nv = 10;
  if (ne > 20) ne = 20;
  print_device_vector<int>(nv, (const int*)d_rowptr_, "row_start");
  print_device_vector<int>(ne, (const int*)d_colidx_, "edge_dst");
}

void LearningGraph::dealloc() {
  uint_free_device(d_rowptr_);
  uint_free_device(d_colidx_);
  if (d_vertex_data_) float_free_device(d_vertex_data_);
  if (d_edge_data_) float_free_device(d_edge_data_);
  if (d_trans_edge_data_) float_free_device(d_trans_edge_data_);
}

