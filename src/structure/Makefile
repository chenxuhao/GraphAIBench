include ../common.mk
OBJS = graph.o VertexSet.o
SIMDCAI = ../../external/SIMDCAI
INCLUDES += -I./gpu_kernels -I$(SIMDCAI)/include
all: compressor tc_omp_compressed tc_gpu_compressed verify_compression

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

compressor: test_cgr_compressor.o cgr_compressor.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) test_cgr_compressor.o cgr_compressor.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

vbyte_compressor: vbyte_compressor.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -L$(SIMDCAI) vbyte_compressor.o $(OBJS) -o $@ -lgomp -lSIMDCompressionAndIntersection
	mv $@ $(BIN)

bfs_gcgt: bfs_main.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) bfs_main.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp_compressed: tc_omp_compressed.o tc_main.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tc_omp_compressed.o tc_main.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

tc_gpu_compressed: tc_gpu_compressed.o tc_main.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tc_gpu_compressed.o tc_main.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp_vbyte: tc_omp_vbyte.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -L$(SIMDCAI) tc_omp_vbyte.o $(OBJS) -o $@ -lgomp -lSIMDCompressionAndIntersection
	mv $@ $(BIN)

verify_compression: verify_compression.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) verify_compression.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

query_graph_info: query_graph_info.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) query_graph_info.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

query_compressed_graph_info: query_compressed_graph_info.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) query_compressed_graph_info.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
