include ../../common.mk
INCLUDES:=-I../../../include -I../
VPATH+=../../common
BIN = ../../../bin
# INCLUDES+=-I$(CILK_CLANG)/include
#OBJS += verifier.o
all: $(OBJS) khop_base

khop_base: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
