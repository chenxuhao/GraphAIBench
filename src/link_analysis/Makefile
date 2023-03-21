include ../common.mk
OBJS += verifier.o
all: pr_omp_base pr_omp_push pr_gpu_base pr_gpu_warp pr_gpu_push

pr_gpu_base: gpu_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

pr_gpu_warp: gpu_warp.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_warp.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

pr_gpu_push: gpu_push.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_push.o -o $@ $(NVLIBS)
	mv $@ $(BIN)

pr_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_push: omp_push.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_push.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
