include ../common.mk
all: test_partitioner
OBJS = VertexSet.o graph.o graph_partition.o test_partitioner.o

test_partitioner: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
