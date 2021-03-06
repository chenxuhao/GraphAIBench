include ../common.mk
INCLUDES+=-I./gpu_kernels
VPATH += ../partitioner
all: tc_omp_base tc_gpu_base tc_multigpu_base tc_multigpu

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

tc_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_omp_simd: $(OBJS) omp_simd.o intersect.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_simd.o intersect.o -o $@ -lgomp
	mv $@ $(BIN)

tc_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_multigpu_base: $(OBJS) multigpu_base.o scheduler.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) multigpu_base.o scheduler.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_multigpu: $(OBJS) multigpu.o scheduler.o graph_partition.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) multigpu.o graph_partition.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_dist_cpu: $(OBJS) dist_cpu.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_cpu.o -o $@ -lgomp
	mv $@ $(BIN)

tc_dist_gpu: $(OBJS) dist_gpu.o gpu_kernel_wrapper.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_gpu.o gpu_kernel_wrapper.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
