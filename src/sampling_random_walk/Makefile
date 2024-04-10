include ../common.mk
OBJS = graph.o VertexSet.o
CGOBJS = graph_compressed.o cgr_decoder.o vbyte_decoder.o
CGCUOBJS = graph_gpu_compressed.o cgr_decoder_gpu.o
COBJS = cgr_encoder.o unary_encoder.o vbyte_encoder.o
NVFLAGS += -dc
CXXFLAGS += -Wno-narrowing
#LIBS += $(SIMDCAI_LIB)
NVINCLUDES += -I./gpu_kernels
INCLUDES +=-I$(CILK_CLANG)/include/cilk
vpath %.cc ../common
vpath %.cc ../structure
vpath %.cu ../structure
# VPATH+=../common
BIN = ../../bin

all: $(OBJS) khop_demo

khop_demo: $(OBJS) $(CGOBJS) $(COBJS) compressor.o khop_demo.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) compressor.o khop_demo.o $(OBJS) $(CGOBJS) $(COBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

rwalk_cpu: $(OBJS) $(CGOBJS) $(COBJS) compressor.o random_walk_demo.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) compressor.o random_walk_cpu.o $(OBJS) $(CGOBJS) $(COBJS) -o $@ $(LIBS)
	mv $@ $(BIN)


clean:
	rm *.o
