include ../common.mk
SRC=main.cu
OBJS += verifier.o
all: $(OBJS) pr_omp_base pr_omp_push pr_gpu_base pr_gpu_warp pr_gpu_push

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

pr_gpu_base: gpu_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_gpu_warp: gpu_warp.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_warp.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_gpu_push: gpu_push.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_push.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_push: omp_push.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_push.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

pr_pb: pb.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o pb.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_pb: omp_pb.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_pb.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_delta: omp_delta.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_delta.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_partition: omp_partition.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_partition.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_sim: omp_sim.o main.o verifier.o
	$(CXX) $(SIMFLAGS) $(INCLUDES) main.o omp_sim.o verifier.o $(M5OP) -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_push_pb: main.o push_pb.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o push_pb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_vector: main.o vector.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o vector.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_lb: main.o lb.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_delta: main.o delta.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o delta.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_fusion: main.o fusion.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o fusion.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_partition: main.o partition.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o partition.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_tiling: tiling.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o tiling.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_push_tile: main.o push_tile.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o push_tile.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_nvgraph: main.o nvgraph.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o nvgraph.o verifier.o -o $@ $(LIBS) -lnvgraph
	mv $@ $(BIN)

clean:
	rm *.o
