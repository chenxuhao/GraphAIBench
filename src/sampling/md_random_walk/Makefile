include ../../common.mk
INCLUDES:=-I../../../include -I../
VPATH+=../../common
BIN = ../../../bin
# INCLUDES+=-I$(CILK_CLANG)/include
#OBJS += verifier.o
all: $(OBJS) md_random_walk_base

md_random_walk_base: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
