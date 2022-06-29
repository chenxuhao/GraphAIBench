include ../common.mk
OBJS += verifier.o

ifneq ($(SHOW_ERROR),0)
NVFLAGS += -DCOMPUTE_ERROR
CXXFLAGS += -DCOMPUTE_ERROR
endif

all: $(OBJS) cf_omp_base cf_gpu_base cf_gpu_vector

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

cf_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) gpu_base.o $(OBJS) -o $@ $(LIBS) 
	mv $@ $(BIN)

cf_gpu_vector: gpu_vector.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) gpu_vector.o $(OBJS) -o $@ $(LIBS) -lgomp
	mv $@ $(BIN)

cf_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
