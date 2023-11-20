include ../../common.mk
INCLUDES:=-I../../../include -I../
VPATH+=../../common
BIN = ../../../bin
INCLUDES+=-I$(CILK_CLANG)/include/cilk
#OBJS += verifier.o
all: $(OBJS) khop_base

khop_base: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

khop_cilk_base: cilk_base.o $(OBJS)
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) cilk_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

khop_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
