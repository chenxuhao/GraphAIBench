include ../common.mk
OBJS = graph.o VertexSet.o graph_compressed.o
LIBS += $(SIMDCAI_LIB)
INCLUDES += -I./gpu_kernels $(SIMDCAI_INC)
all: cgr_compressor vbyte_compressor tc_omp_compressed tc_gpu_compressed

hybrid_compressor: hybrid_compressor.o unary_encoder.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) hybrid_compressor.o unary_encoder.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

cgr_compressor: compressor.o cgr_encoder.o unary_encoder.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) cgr_encoder.o compressor.o unary_encoder.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

vbyte_compressor: vbyte_compressor.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -L$(SIMDCAI_HOME) vbyte_compressor.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp_compressed: tc_omp_compressed.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tc_omp_compressed.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

tc_cilk_vbyte: $(OBJS) tc_cilk_vbyte.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) tc_cilk_vbyte.o -o $@
	mv $@ $(BIN)

tc_gpu_compressed: tc_gpu_compressed.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tc_gpu_compressed.o $(OBJS) -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

verify_compression: verify_compression.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) verify_compression.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

query_graph_info: query_graph_info.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) query_graph_info.o $(OBJS) -o $@ $(LIBS) 
	mv $@ $(BIN)

query_compressed_graph_info: query_compressed_graph_info.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) query_compressed_graph_info.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
