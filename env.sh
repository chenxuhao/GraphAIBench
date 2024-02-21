#export LD_LIBRARY_PATH=/usr/local/OpenBLAS/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/software/cuda/11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export KMP_AFFINITY=scatter
export KMP_LIBRARY=turnaround
export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=32

export GPU_ARCH=70
export CUDA_HOME=/software/cuda/11.4
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

export GCC_HOME=/usr/bin/gcc
export CILK_HOME=/projects/bbof/mcai1/OpenCilk/build
export CILK_CLANG=/projects/bbof/mcai1/OpenCilk/build/lib/clang/14.0.6

export DATASET_PATH=/home/cxh/datasets/automine/
