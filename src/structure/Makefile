include ../common.mk
OBJS = graph.o VertexSet.o graph_compressed.o
LIBS += $(SIMDCAI_LIB)
INCLUDES += -I./gpu_kernels $(SIMDCAI_INC)
all: compressor tc_omp_compressed tc_gpu_compressed

compressor: compressor.o cgr_encoder.o unary_encoder.o hybrid_encoder.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) compressor.o unary_encoder.o cgr_encoder.o hybrid_encoder.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp_compressed: cgr_decoder.o tc_omp_compressed.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) cgr_decoder.o tc_omp_compressed.o $(OBJS) -o $@ $(LIBS)
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
