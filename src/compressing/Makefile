include ../common.mk
OBJS = graph.o VertexSet.o
CGOBJS = graph_compressed.o cgr_decoder.o vbyte_decoder.o
CGCUOBJS = graph_gpu_compressed.o cgr_decoder_gpu.o
COBJS = cgr_encoder.o unary_encoder.o vbyte_encoder.o
# CGCUOBJS = graph_gpu_compressed.o cgr_decoder_gpu.o
NVFLAGS += -dc
CXXFLAGS += -Wno-narrowing
#LIBS += $(SIMDCAI_LIB)
INCLUDES += -I./gpu_kernels #-I$(CILK_CLANG)/include/cilk #$(SIMDCAI_INC)
vpath %cc ../common
vpath %cc ../structure
vpath %cu ../structure
# VPATH+=../common
BIN = ../../bin
all: cpu_serial_khop cpu_omp_khop on_gpu_khop unified_gpu_khop

cpu_serial_khop: $(OBJS) $(CGOBJS) $(COBJS) compressor.o cpu_serial_khop.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) compressor.o cpu_serial_khop.o $(OBJS) $(CGOBJS) $(COBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

cpu_omp_khop: $(OBJS) $(CGOBJS) $(COBJS) compressor.o cpu_omp_khop.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) compressor.o cpu_omp_khop.o $(OBJS) $(CGOBJS) $(COBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

test: $(OBJS) $(CGOBJS) $(COBJS) test.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) test.o $(OBJS) $(CGOBJS) $(COBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

save_subgraphs: $(OBJS) $(CGOBJS) $(COBJS) save_subgraphs.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) save_subgraphs.o $(OBJS) $(CGOBJS) $(COBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

gpu_cgr: $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) compressor.o gpu_cgr.o
	$(NVCC) $(CUDA_ARCH) -O3 -w $(INCLUDES) compressor.o gpu_cgr.o $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

gpu_vbyte: $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) compressor.o gpu_vbyte.o
	$(NVCC) $(CUDA_ARCH) -O3 -w $(INCLUDES) compressor.o gpu_vbyte.o $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

gpu_vbyte_warp: $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) compressor.o gpu_vbyte_warp.o
	$(NVCC) $(CUDA_ARCH) -O3 -w $(INCLUDES) compressor.o gpu_vbyte_warp.o $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

on_gpu_khop: $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) compressor.o on_gpu_khop.o
	$(NVCC) $(CUDA_ARCH) -O3 -w $(INCLUDES) compressor.o on_gpu_khop.o $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

gpu_uncomp: $(OBJS) gpu_uncomp.o
	$(NVCC) $(CUDA_ARCH) -O3 -w $(INCLUDES) gpu_uncomp.o $(OBJS) -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

test_decode: $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) compressor.o test_decode.o
	$(NVCC) $(CUDA_ARCH) -O3 -w $(INCLUDES) compressor.o test_decode.o $(OBJS) $(CGOBJS) $(CGCUOBJS) $(COBJS) -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

compressor: $(OBJS) compressor.o cgr_encoder.o unary_encoder.o vbyte_encoder.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) compressor.o unary_encoder.o cgr_encoder.o vbyte_encoder.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

main_clean:
	rm cpu_serial_khop.o cpu_omp_khop.o on_gpu_khop.o unified_gpu_khop.o
