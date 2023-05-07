DEBUG ?= 0
BIN = ../../bin/
CC := gcc
CXX := g++
ICC := $(ICC_HOME)/icc
ICPC := $(ICC_HOME)/icpc
MPICC := mpicc
MPICXX := mpicxx
NVCC := nvcc
#NVCC := $(CUDA_HOME)/bin/nvcc
CLANG := $(CILK_HOME)/bin/clang
CLANGXX := $(CILK_HOME)/bin/clang++
SIMDCAI_HOME := ../../external/SIMDCAI

UNAME_P := $(shell uname -p)
ifndef GPU_ARCH
GPU_ARCH = 80
endif
CUDA_ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
CXXFLAGS  := -Wall -fopenmp -std=c++17
#ifneq ($(UNAME_P), arm)
ifeq ($(UNAME_P), x86_64)
  CXXFLAGS += -march=native
endif
ICPCFLAGS := -O3 -Wall -qopenmp
NVFLAGS := $(CUDA_ARCH)
NVFLAGS += -Xptxas -v -std=c++17
NVFLAGS += -DUSE_GPU
NVLIBS = -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcuda -lcudart
MPI_LIBS = -L$(MPI_HOME)/lib -lmpi
NVSHMEM_LIBS = -L$(NVSHMEM_HOME)/lib -lnvshmem -lnvToolsExt -lnvidia-ml -ldl -lrt
CILKFLAGS=-O3 -fopenmp=libiomp5 -fopencilk
CILK_INC=-I$(GCC_HOME)/include -I$(CILK_CLANG)/include
SIMDCAI_LIB := -L$(SIMDCAI_HOME) -lSIMDCompressionAndIntersection
SIMDCAI_INC := -I$(SIMDCAI_HOME)/include
INCLUDES := -I../../include
LIBS := -lgomp

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

ifeq ($(PAPI), 1)
CXXFLAGS += -DENABLE_PAPI
INCLUDES += -I$(PAPI_HOME)/include
LIBS += -L$(PAPI_HOME)/lib -lpapi
endif

ifeq ($(USE_TBB), 1)
LIBS += -L$(TBB_HOME)/lib/intel64/gcc4.8/ -ltbb
endif

VPATH += ../common
OBJS=main.o VertexSet.o graph.o

ifneq ($(NVSHMEM),)
CXXFLAGS += -DUSE_MPI
NVFLAGS += -DUSE_NVSHMEM -DUSE_MPI -dc
endif

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
else
  NVFLAGS += -DVERTEX_PAR
endif

# CUDA unified memory
ifneq ($(USE_UM),)
NVFLAGS += -DUSE_UM
endif

# kernel fission
ifneq ($(FISSION),)
NVFLAGS += -DFISSION
endif

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

%.o: %.cxx
	$(CLANGXX) $(CILKFLAGS) $(INCLUDES) $(CILK_INC) -c $<

