include ../common.mk
#VPATH += ../partitioner
INCLUDES+=-I./gpu_kernels -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include
all: tc_omp_base tc_gpu_base tc_multigpu_base

tc_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_omp_simd: $(OBJS) omp_simd.o intersect.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_simd.o intersect.o -o $@ -lgomp
	mv $@ $(BIN)

tc_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

tc_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

tc_gpu_hindex: $(OBJS) gpu_hindex.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_hindex.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

tc_multigpu_base: $(OBJS) multigpu_base.o scheduler.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) multigpu_base.o scheduler.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

tc_multigpu_induced: $(OBJS) multigpu_induced.o graph_partition.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) multigpu_induced.o graph_partition.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

tc_multigpu_nvshmem: $(OBJS) multigpu_nvshmem.o graph_partition.o
	$(NVCC) $(CUDA_ARCH) -O3 -w -DUSE_NVSHMEM -DUSE_MPI $(INCLUDES) $(OBJS) multigpu_nvshmem.o graph_partition.o -o $@ $(NVSHMEM_LIBS) $(NVLIBS) $(MPI_LIBS)
	mv $@ $(BIN)

tc_dist_cpu: $(OBJS) dist_cpu.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_cpu.o -o $@ -lgomp
	mv $@ $(BIN)

tc_dist_gpu: $(OBJS) dist_gpu.o gpu_kernel_wrapper.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_gpu.o gpu_kernel_wrapper.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

clean:
	rm *.o
