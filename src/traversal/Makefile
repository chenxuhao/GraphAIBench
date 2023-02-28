include ../common.mk
OBJS += verifier.o 
all: bfs_omp_direction bfs_omp_base bfs_gpu_base bfs_gpu_twc sssp_omp_base sssp_gpu_base sssp_gpu_twc

bfs_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_topo: omp_topo.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_topo.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_direction: omp_direction.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_direction.o -o $@ -lgomp
	mv $@ $(BIN)

# data-driven BFS baseline
bfs_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

# data-driven BFS using TWC load balancing
bfs_gpu_twc: gpu_twc.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_twc.o -o $@ $(LIBS)
	mv $@ $(BIN)

sssp_omp_base: omp_dstep.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_dstep.o -o $@ -lgomp
	mv $@ $(BIN)

sssp_gpu_base: gpu_bellmanford.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_bellmanford.o -o $@ $(LIBS)
	mv $@ $(BIN)

sssp_gpu_twc: gpu_bellmanford_twc.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_bellmanford_twc.o -o $@ $(LIBS)
	mv $@ $(BIN)

sssp_gpu_dstep: gpu_dstep.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_dstep.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
