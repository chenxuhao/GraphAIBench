include ../common.mk
OBJS = VertexSet.o graph.o graph_partition.o
all: test_graph_partition test_nvlink

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $<

test_graph_partition: $(OBJS) test_graph_partition.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) test_graph_partition.o -o $@ -lgomp
	mv $@ $(BIN)

test_nvlink: 
	$(NVCC) $(NVFLAGS) $(INCLUDES) test_nvlink.cu -o test_nvlink -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
