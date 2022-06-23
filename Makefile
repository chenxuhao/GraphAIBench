KERNELS = centrality components link_analysis triangle traversal

.PHONY: all
all: $(KERNELS)

% : src/%/Makefile
	cd src/$@; make; cd ../..

.PHONY: clean
clean:
	rm src/*/*.o
