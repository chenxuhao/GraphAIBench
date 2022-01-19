DEBUG ?= 0
USE_GPU ?= 0
ENABLE_TILING ?= 0
CC=g++
NVCC=nvcc
IDIR=./include
ODIR=./obj
CUDA_HOME=/usr/local/cuda
#CUDA_HOME=/org/centers/cdgc/cuda/cuda-10.2
#CUDA_HOME=/jet/packages/spack/opt/spack/linux-centos8-zen/gcc-8.3.1/cuda-10.2.89-kz7u4ix6ed53nioz4ycqin3kujcim3bs
OPENBLAS_DIR=/usr/local/OpenBLAS/build
#OPENBLAS_DIR=/org/centers/cdgc/openblas/ubuntu-gcc7.5
#OPENBLAS_DIR=/org/centers/cdgc/openblas/centos-gcc9.2
#OPENBLAS_DIR=/ocean/projects/cie170003p/shared/OpenBLAS/build
CUB_DIR=../cub
#CUB_DIR=/ocean/projects/cie170003p/shared/cub
MKL_DIR=/opt/apps/sysnet/intel/20.0/mkl
CFLAGS=-I./include -fopenmp -pthread -Wall --std=c++11 -lboost_thread -lboost_system -I$(CUDA_HOME)/include
CUFLAGS=-I./include -DENABLE_GPU -I${CUB_DIR} -lcudart -lcublas -lcusparse -lcurand -lboost_thread -lboost_system -I$(CUDA_HOME)/include

ifeq ($(DEBUG), 1)
	CFLAGS += -g -O0
	CUFLAGS += -G -lineinfo
else
	CFLAGS += -O3
	CUFLAGS += -O3 -w
endif

ifeq ($(ENABLE_TILING), 1)
	CFLAGS += -DCSR_SEGMENTING
endif

#CFLAGS += -mavx512f -march=native

_DEPS=global.h configs.h # global dependencies
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ=reader.o loss_layer.o net.o sampler.o random.o l2norm_layer.o dense_layer.o

ifeq ($(GNN_SAGE), 1)
	CFLAGS += -DUSE_SAGE
	CUFLAGS += -DUSE_SAGE
endif

ifeq ($(GNN_GAT), 1)
	CFLAGS += -DUSE_GAT
	CUFLAGS += -DUSE_GAT
endif

ifeq ($(USE_GPU), 1)
	CFLAGS += -DENABLE_GPU
endif

ifeq ($(USE_MKL), 1)
	CFLAGS += -DUSE_MKL -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -I${MKL_DIR}/include -L${MKL_DIR}/lib/intel64 -liomp5
else
	CFLAGS += -I${OPENBLAS_DIR}/include -L$(OPENBLAS_DIR)/lib -lopenblas
endif

ifeq ($(USE_GPU), 1)
	_OBJ += graph_conv_layer.cu.o lgraph.cu.o math_functions.cu.o optimizer.cu.o softmax_loss_layer.cu.o sigmoid_loss_layer.cu.o gcn_layer.cu.o gcn_aggregator.cu.o gat_layer.cu.o gat_aggregator.cu.o sage_layer.cu.o sage_aggregator.cu.o
else
	_OBJ += graph_conv_layer.o lgraph.o math_functions.o optimizer.o softmax_loss_layer.o sigmoid_loss_layer.o gcn_layer.o gcn_aggregator.o gat_layer.o gat_aggregator.o sage_layer.o sage_aggregator.o
endif

OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.cu.o: src/%.cu $(IDIR)/%.h $(DEPS)
	@mkdir -p $(@D)
	$(NVCC) -c -o $@ $< $(CUFLAGS)

$(ODIR)/%.cu.o: src/gconv/%.cu $(IDIR)/graph_conv_layer.h $(DEPS)
	@mkdir -p $(@D)
	$(NVCC) -c -o $@ $< $(CUFLAGS)

$(ODIR)/%.o: src/%.cpp $(IDIR)/%.h $(DEPS)
	@mkdir -p $(@D)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: src/gconv/%.cpp $(IDIR)/graph_conv_layer.h $(DEPS)
	@mkdir -p $(@D)
	$(CC) -c -o $@ $< $(CFLAGS)

cpu_train_gcn: src/train.cpp $(OBJ)
	$(CC) src/train.cpp -o bin/$@ $(OBJ) $(CFLAGS) $(LIBS)

cpu_train_sage: src/train.cpp $(OBJ)
	$(CC) src/train.cpp -o bin/$@ $(OBJ) $(CFLAGS) $(LIBS)

cpu_train_gat: src/train.cpp $(OBJ)
	$(CC) src/train.cpp -o bin/$@ $(OBJ) $(CFLAGS) $(LIBS)

gpu_train_gcn: src/train.cpp $(OBJ)
	$(NVCC) src/train.cpp -o bin/$@ $(OBJ) $(CUFLAGS) $(LIBS) -lgomp

gpu_train_sage: src/train.cpp $(OBJ)
	$(NVCC) src/train.cpp -o bin/$@ $(OBJ) $(CUFLAGS) $(LIBS) -lgomp

gpu_train_gat: src/train.cpp $(OBJ)
	$(NVCC) src/train.cpp -o bin/$@ $(OBJ) $(CUFLAGS) $(LIBS) -lgomp

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o

