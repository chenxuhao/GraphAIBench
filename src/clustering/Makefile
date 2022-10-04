include ../common.mk
INCLUDES += -I../../external/PAM/include -I../../external/parlaylib/include
OBJS = VertexSet.o graph.o
all: hac_serial

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

hac_serial: $(OBJS) main.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) main.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
