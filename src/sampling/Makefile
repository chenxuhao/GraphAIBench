include ../common.mk
#OBJS += verifier.o
all: $(OBJS) cpu_khop gpu_khop omp_khop

sample_gpu_base: gpu_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

sample_gpu_warp: gpu_warp.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_warp.o -o $@ $(LIBS)
	mv $@ $(BIN)

sample_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

gpu_khop: gpu_khop.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_khop.o -o $@ $(LIBS)
	mv $@ $(BIN)

cpu_khop: cpu_khop.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) cpu_khop.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

omp_khop: omp_khop.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_khop.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
