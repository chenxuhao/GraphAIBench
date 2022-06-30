include ../common.mk
OBJS += verifier.o
all: cc_omp_base cc_omp_afforest cc_gpu_base cc_gpu_warp cc_gpu_afforest

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

cc_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

cc_omp_afforest: omp_afforest.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_afforest.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

cc_gpu_base: gpu_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_gpu_afforest: gpu_afforest.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_afforest.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_gpu_warp: gpu_warp.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_warp.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
