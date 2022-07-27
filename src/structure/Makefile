include ../common.mk
all: cgr_compressor

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

cgr_compressor: cgr_compressor.o graph.o VertexSet.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) cgr_compressor.o graph.o VertexSet.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_gcgt: bfs_main.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) bfs_main.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
