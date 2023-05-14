#export LD_LIBRARY_PATH=/usr/local/OpenBLAS/build/lib:$LD_LIBRARY_PATH
export KMP_AFFINITY=scatter
export KMP_LIBRARY=turnaround
export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=32

export GPU_ARCH=80
export CUDA_HOME=/usr/local/cuda
#export CUDA_HOME=/opt/apps/cuda/11.3
# export CUDA_HOME=/data/apps/linux-centos8-cascadelake/gcc-9.3.0/cuda-11.1.0-ojwpqodqcxvz4meuw2skpy6riar6ni7d
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_MPI_SUPPORT=1
#export NVSHMEM_SHMEM_SUPPORT=1
export MPI_HOME=/usr
export NVSHMEM_PREFIX=/usr/local/nvshmem
export NVSHMEM_HOME=/usr/local/nvshmem
#PAPI_HOME = /usr/local/papi-6.0.0
#ICC_HOME = /opt/intel/compilers_and_libraries/linux/bin/intel64
export OPENBLAS_DIR=/usr/local/openblas
export MKL_DIR=/opt/apps/sysnet/intel/20.0/mkl
#MKLROOT = /opt/intel/mkl

export GCC_HOME=/usr/lib/gcc/x86_64-linux-gnu/8
export CILK_HOME=/home/cxh/OpenCilk/build
export CILK_CLANG=/home/cxh/OpenCilk/build/lib/clang/14.0.6

