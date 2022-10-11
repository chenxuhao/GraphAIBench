include ../common.mk
OBJS = graph.o VertexSet.o
all: converter cleaner

%.o: %.cc
	g++ $(CXXFLAGS) $(INCLUDES) -fopenmp -I../include -c $<

converter: $(OBJS) converter.o main.o
	g++ $(CXXFLAGS) $(INCLUDES) $(OBJS) converter.o main.o -o $@ -lgomp
	mv $@ $(BIN)

cleaner: $(OBJS) cleaner.o
	g++ $(CXXFLAGS) $(INCLUDES) $(OBJS) cleaner.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
