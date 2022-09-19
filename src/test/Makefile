include ../common.mk
OBJS = VertexSet.o graph.o graph_partition.o
NVFLAGS += -dc
INCLUDES += -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include
all: test_graph_partition test_nvlink test_cta_sort

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $<

test_graph_partition: $(OBJS) test_graph_partition.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) test_graph_partition.o -o $@ -lgomp
	mv $@ $(BIN)

test_cta_sort: 
	$(NVCC) -O3 $(INCLUDES) test_cta_sort.cu -o test_cta_sort
	mv $@ $(BIN)

test_nvlink: 
	$(NVCC) $(NVFLAGS) $(INCLUDES) test_nvlink.cu -o test_nvlink -lgomp
	mv $@ $(BIN)

test_nvshmem: test_nvshmem.o
	$(NVCC) $(CUDA_ARCH) $(INCLUDES) test_nvshmem.o -o $@ $(NVSHMEM_LIBS) $(LIBS)
	mv $@ $(BIN)

test_nvshmem_mpi: test_nvshmem_mpi.o
	$(NVCC) $(CUDA_ARCH) $(INCLUDES) test_nvshmem_mpi.o -o $@ $(NVSHMEM_LIBS) $(LIBS) $(MPI_LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
