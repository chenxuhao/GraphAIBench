include ../common.mk
OBJS += verifier.o

all: $(OBJS) cc_omp_base cc_omp_afforest

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

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

cc_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) main.o ocl_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

cc_base: main.o base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_afforest: main.o afforest.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o afforest.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_partition: main.o partition.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o partition.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_warp: main.o warp.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o warp.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_fusion: main.o fusion.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o fusion.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
