include ../../common.mk
INCLUDES:=-I../../../include -I../
VPATH+=../../common
BIN = ../../../bin
# INCLUDES+=-I$(CILK_CLANG)/include
#OBJS += verifier.o
all: $(OBJS) node2vec_base

node2vec_base: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
