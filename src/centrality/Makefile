include ../common.mk
OBJS += verifier.o
all: bc_omp_base bc_gpu_base bc_gpu_twc

bc_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

# data driven baseline
bc_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@  $(BIN)

# TWC load balancing
bc_gpu_twc: gpu_twc.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_twc.o -o $@ $(NVLIBS)
	mv $@  $(BIN)

# topology driven baseline
bc_topo_base: topo_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) topo_base.o -o $@ $(NVLIBS)
	mv $@  $(BIN)

# topology driven + TWC load balancing
bc_topo_twc: topo_twc.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) topo_twc.o -o $@ $(NVLIBS)
	mv $@  $(BIN)

clean:
	rm *.o
