include ../common.mk
INCLUDES += -I../../external/PAM/include -I../../external/parlaylib/include
OBJS = VertexSet.o graph.o
all: hac_serial

hac_serial: $(OBJS) main.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) main.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o
