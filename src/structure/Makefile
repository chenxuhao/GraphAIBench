include ../common.mk
OBJS = graph.o VertexSet.o
all: test_cgr_compressor bfs_gcgt tc_omp_compressed

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

test_cgr_compressor: test_cgr_compressor.o cgr_compressor.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) test_cgr_compressor.o cgr_compressor.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

bfs_gcgt: bfs_main.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) bfs_main.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp_compressed: tc_omp_compressed.o tc_main.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tc_omp_compressed.o tc_main.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
