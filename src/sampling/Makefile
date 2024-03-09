include ../common.mk
INCLUDES:=-I../../include -I../
VPATH+=../common
BIN = ../../bin
# INCLUDES+=-I$(CILK_CLANG)/include/cilk
#OBJS += verifier.o
all: n2v khop $(OBJS)

n2v: n2v_serial n2v_omp

khop: khop_serial khop_omp

n2v_serial: n2v_serial.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) n2v_serial.o -o $@ -lgomp
	mv $@ $(BIN)

n2v_omp: n2v_omp.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) n2v_omp.o -o $@ -lgomp
	mv $@ $(BIN)

khop_serial: khop_serial.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) khop_serial.o -o $@ -lgomp
	mv $@ $(BIN)

khop_omp: khop_omp.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) khop_omp.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
