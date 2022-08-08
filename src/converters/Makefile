include ../common.mk
OBJS = converter.o graph.o VertexSet.o main.o
all: converter

%.o: %.cc
	g++ $(CXXFLAGS) $(INCLUDES) -fopenmp -I../include -c $<

converter: $(OBJS)
	g++ $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $@ -lgomp

clean:
	rm *.o converter
