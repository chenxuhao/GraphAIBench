include ../../common.mk
INCLUDES:=-I../../../include -I../
VPATH+=../../common
BIN = ../../../bin
INCLUDES+=-I$(CILK_CLANG)/include/cilk
#OBJS += verifier.o
all: $(OBJS) importance_omp_base

importance_base: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

importance_cilk_base: cilk_base.o $(OBJS)
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) cilk_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

importance_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
