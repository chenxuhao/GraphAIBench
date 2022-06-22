DEBUG ?= 0
USE_DRAMSIM3 ?= 1
CUDA_HOME=/usr/local/cuda
PAPI_HOME=/usr/local/papi-6.0.0
ICC_HOME=/opt/intel/compilers_and_libraries/linux/bin/intel64
MKLROOT=/opt/intel/mkl
CUB_DIR=../../../cub
MGPU_DIR=../../../moderngpu
BIN=../../bin/
HOST=X86
ifeq ($(HOST),X86)
CC=gcc
CXX=g++
else 
CC=aarch64-linux-gnu-gcc
CXX=aarch64-linux-gnu-g++
endif
ICC=$(ICC_HOME)/icc
ICPC=$(ICC_HOME)/icpc
MPICC=mpicc
MPICXX=mpicxx
NVCC=nvcc
#NVCC=$(CUDA_HOME)/bin/nvcc
CUDA_ARCH := \
	-gencode arch=compute_61,code=sm_61
CXXFLAGS=-Wall -fopenmp -std=c++17 -march=native
ICPCFLAGS=-O3 -Wall -qopenmp
NVFLAGS=$(CUDA_ARCH)
NVFLAGS+=-Xptxas -v
NVFLAGS+=-DUSE_GPU
ifeq ($(HOST),X86)
#SIMFLAGS=-O3 -Wall -DSIM -fopenmp -static -L/home/cxh/m5threads/ -lpthread
SIMFLAGS=-Wall -fopenmp -std=c++17 -O3
else
SIMFLAGS=-flto -fwhole-program -O3 -Wall -fopenmp -static
endif
#M5OP=/home/cxh/gem5/util/m5/src/arm/m5op.S
M5OP=/home/cxh/gem5/util/m5/src/x86/m5op.S

ifeq ($(VTUNE), 1)
	CXXFLAGS += -g
endif
ifeq ($(NVPROF), 1)
	NVFLAGS += -lineinfo
endif

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G
else
	CXXFLAGS += -O3
	NVFLAGS += -O3 -w
endif

INCLUDES = -I../../include
LIBS=-L$(CUDA_HOME)/lib64 -lcudart -lgomp

ifeq ($(PAPI), 1)
CXXFLAGS += -DENABLE_PAPI
INCLUDES += -I$(PAPI_HOME)/include
LIBS += -L$(PAPI_HOME)/lib -lpapi
endif

ifeq ($(USE_TBB), 1)
LIBS += -L/h2/xchen/work/gardenia_code/tbb2020/lib/intel64/gcc4.8/ -ltbb
endif

ifeq ($(SIM), 1)
CXXFLAGS=-DSIM $(SIMFLAGS) 
EXTRA=$(M5OP)
INCLUDES+=-I/home/cxh/gem5/include
LIBS += -pthread -lrt -ldl
endif

VPATH += ../common
OBJS=main.o VertexSet.o graph.o

# CUDA vertex parallel
ifneq ($(VPAR),)
NVFLAGS += -DVERTEX_PAR
endif

# CUDA CTA centric
ifneq ($(CTA),)
NVFLAGS += -DCTA_CENTRIC
endif

ifneq ($(PROFILE),)
CXXFLAGS += -DPROFILING
endif

ifneq ($(USE_SET_OPS),)
CXXFLAGS += -DUSE_MERGE
endif

ifneq ($(USE_SIMD),)
CXXFLAGS += -DSI=0
endif

# counting or listing
ifneq ($(COUNT),)
NVFLAGS += -DDO_COUNT
endif

# GPU vertex/edge parallel 
ifeq ($(VERTEX_PAR),)
NVFLAGS += -DEDGE_PAR
endif

# CUDA unified memory
ifneq ($(USE_UM),)
NVFLAGS += -DUSE_UM
endif

# kernel fission
ifneq ($(FISSION),)
NVFLAGS += -DFISSION
endif

