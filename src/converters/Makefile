include ../common.mk
all: mtx2bin

%.o: %.cc
	g++ $(CXXFLAGS) $(INCLUDES) -fopenmp -I../include -c $<

mtx2bin: mtx2bin.o converter.o
	g++ $(CXXFLAGS) $(INCLUDES) mtx2bin.o converter.o -o mtx2bin -lgomp

clean:
	rm *.o
