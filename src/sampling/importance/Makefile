include ../../common.mk
INCLUDES:=-I../../../include -I../
VPATH+=../../common
BIN = ../../../bin
# INCLUDES+=-I$(CILK_CLANG)/include
#OBJS += verifier.o
all: $(OBJS) importance_base

importance_base: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
