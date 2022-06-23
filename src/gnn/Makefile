DEBUG ?= 0
USE_GPU ?= 0
ENABLE_TILING ?= 0
CC=g++
NVCC=nvcc
IDIR=../../include/gnn
ODIR=./obj
CUDA_HOME=/usr/local/cuda
#CUDA_HOME=/org/centers/cdgc/cuda/cuda-10.2
#CUDA_HOME=/jet/packages/spack/opt/spack/linux-centos8-zen/gcc-8.3.1/cuda-10.2.89-kz7u4ix6ed53nioz4ycqin3kujcim3bs
#OPENBLAS_DIR=/usr/local/openblas
#OPENBLAS_DIR=/org/centers/cdgc/openblas/ubuntu-gcc7.5
#OPENBLAS_DIR=/org/centers/cdgc/openblas/centos-gcc9.2
OPENBLAS_DIR=/ocean/projects/cie170003p/shared/OpenBLAS/build
MKL_DIR=/opt/apps/sysnet/intel/20.0/mkl
CFLAGS=-I../../include -I../../include/gnn -I../../include/layers -I../../include/utils -fopenmp -pthread -Wall --std=c++11 -lboost_thread -lboost_system -I$(CUDA_HOME)/include
CUFLAGS=-I../../include -I../../include/gnn -I../../include/layers -I../../include/utils -DENABLE_GPU -I${OPENBLAS_DIR}/include -lcudart -lcublas -lcusparse -lcurand -lboost_thread -lboost_system -I$(CUDA_HOME)/include

VPATH += ../common
VPATH += ../utilities
VPATH += ../layers
VPATH += ./gconv

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

GNN_LAYER_CUOBJS = gcn_layer.cu.o gcn_aggregator.cu.o gat_layer.cu.o gat_aggregator.cu.o sage_layer.cu.o sage_aggregator.cu.o
GNN_LAYER_CXXOBJS = gcn_layer.o gcn_aggregator.o gat_layer.o gat_aggregator.o sage_layer.o sage_aggregator.o
LAYER_CUOBJS = softmax_loss_layer.cu.o sigmoid_loss_layer.cu.o
LAYER_CXXOBJS = softmax_loss_layer.o sigmoid_loss_layer.o
COMMON_CXXOBJS = graph_conv_layer.o lgraph.o
COMMON_CUOBJS = graph_conv_layer.cu.o lgraph.cu.o 
UTIL_CUOBJS = math_functions.cu.o optimizer.cu.o
UTIL_CXXOBJS = math_functions.o optimizer.o
LAYER_OBJS = l2norm_layer.o dense_layer.o
COMMON_OBJS = reader.o loss_layer.o net.o sampler.o random.o

ifeq ($(USE_GPU), 1)
  LAYER_OBJS += $(LAYER_CUOBJS)
  GNN_LAYER_OBJS = $(GNN_LAYER_CUOBJS)
  COMMON_OBJS = $(COMMON_CUOBJS)
  UTIL_OBJS = $(UTIL_CUOBJS)
else
  LAYER_OBJS += $(LAYER_CXXOBJS)
  GNN_LAYER_OBJS = $(GNN_LAYER_CXXOBJS)
  COMMON_OBJS = $(COMMON_CXXOBJS)
  UTIL_OBJS = $(UTIL_CXXOBJS)
endif

OBJS = $(LAYER_OBJS) $(GNN_LAYER_OBJS_) $(COMMON_OBJS) $(UTIL_OBJS)
OBJ = $(patsubst %,$(ODIR)/%,$(OBJS))

$(ODIR)/%.cu.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) -c -o $@ $< $(CUFLAGS)

$(ODIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CC) -c -o $@ $< $(CFLAGS)

BIN_DIR=../../bin

cpu_train_gcn: train.cpp $(OBJ)
	$(CC) train.cpp -o $(BIN_DIR)/$@ $(OBJ) $(CFLAGS) $(LIBS)

cpu_train_sage: train.cpp $(OBJ)
	$(CC) train.cpp -o $(BIN_DIR)/$@ $(OBJ) $(CFLAGS) $(LIBS)

cpu_train_gat: train.cpp $(OBJ)
	$(CC) train.cpp -o $(BIN_DIR)/$@ $(OBJ) $(CFLAGS) $(LIBS)

gpu_train_gcn: train.cpp $(OBJ)
	$(NVCC) train.cpp -o $(BIN_DIR)/$@ $(OBJ) $(CUFLAGS) $(LIBS) -lgomp

gpu_train_sage: train.cpp $(OBJ)
	$(NVCC) train.cpp -o $(BIN_DIR)/$@ $(OBJ) $(CUFLAGS) $(LIBS) -lgomp

gpu_train_gat: train.cpp $(OBJ)
	$(NVCC) train.cpp -o $(BIN_DIR)/$@ $(OBJ) $(CUFLAGS) $(LIBS) -lgomp

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o

