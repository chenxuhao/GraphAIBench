include ../common.mk
OBJS = graph.o VertexSet.o
SIMDCAI_HOME := ../../external/SIMDCAI
SIMDCAI_LIB := -L$(SIMDCAI_HOME) -lSIMDCompressionAndIntersection
SIMDCAI_INC := -I$(SIMDCAI_HOME)/include
INCLUDES += -I./gpu_kernels $(SIMDCAI_INC)
all: cgr_compressor vbyte_compressor tc_omp_compressed tc_gpu_compressed

cgr_compressor: test_cgr_compressor.o cgr_compressor.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) test_cgr_compressor.o cgr_compressor.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

vbyte_compressor: vbyte_compressor.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -L$(SIMDCAI_HOME) vbyte_compressor.o $(OBJS) -o $@ -lgomp -lSIMDCompressionAndIntersection
	mv $@ $(BIN)

tc_omp_compressed: tc_omp_compressed.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tc_omp_compressed.o $(OBJS) -o $@ -lgomp $(SIMDCAI_LIB)
	mv $@ $(BIN)

tc_cilk_vbyte: $(OBJS) tc_cilk_vbyte.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) tc_cilk_vbyte.o -o $@
	mv $@ $(BIN)

tc_gpu_compressed: tc_gpu_compressed.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tc_gpu_compressed.o $(OBJS) -o $@ $(LIBS)
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

bfs_gcgt: bfs_main.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) bfs_main.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
