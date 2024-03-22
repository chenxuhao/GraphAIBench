include ../common.mk
#OBJS += verifier.o
OBJS = graph.o VertexSet.o

all: $(OBJS) khop_demo

khop_demo: khop_demo.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) khop_demo.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
