include ../common.mk
OBJS = graph.o VertexSet.o
CGOBJS = graph_compressed.o cgr_decoder.o vbyte_decoder.o
CGCUOBJS = graph_gpu_compressed.o cgr_decoder_gpu.o
NVFLAGS += -dc
CXXFLAGS += -Wno-narrowing
LIBS += $(SIMDCAI_LIB)
INCLUDES += -I./gpu_kernels $(SIMDCAI_INC)
all: compressor tc_omp_compressed tc_gpu_compressed

compressor: $(OBJS) compressor.o cgr_encoder.o unary_encoder.o vbyte_encoder.o hybrid_encoder.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) compressor.o unary_encoder.o cgr_encoder.o vbyte_encoder.o hybrid_encoder.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp_compressed: tc_omp_compressed.o $(OBJS) $(CGOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tc_omp_compressed.o $(OBJS) $(CGOBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

tc_cilk_vbyte: $(OBJS) $(CGOBJS) tc_cilk_vbyte.o 
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) $(OBJS) $(CGOBJS) tc_cilk_vbyte.o -o $@
	mv $@ $(BIN)

tc_gpu_compressed: $(OBJS) $(CGOBJS) $(CGCUOBJS) tc_gpu_compressed.o
	$(NVCC) $(CUDA_ARCH) -O3 -w $(INCLUDES) tc_gpu_compressed.o $(OBJS) $(CGOBJS) $(CGCUOBJS) -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)
#	$(CXX) $(CXXFLAGS) $(INCLUDES) tc_gpu_compressed.o $(OBJS) $(CGOBJS) $(CGCUOBJS) -o $@ $(LIBS) $(NVLIBS)

verify_compression: verify_compression.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) verify_compression.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

query_graph_info: query_graph_info.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) query_graph_info.o $(OBJS) -o $@ $(LIBS) 
	mv $@ $(BIN)

query_compressed_graph_info: query_compressed_graph_info.o $(OBJS) $(CGOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) query_compressed_graph_info.o $(OBJS) $(CGOBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
