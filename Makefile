KERNELS = centrality components embedding link_analysis sampling triangle traversal

.PHONY: all
all: $(KERNELS)

% : src/%/Makefile
	cd src/$@; make; cd ../..

.PHONY: clean
clean:
	rm src/*/*.o
	rm src/gnn/obj/*.o
