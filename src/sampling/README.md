# graph sampling
## Usage
For gpu implementation, if you want to run livej with batch size=40000 and #cuda threads=40000, you can
```
make -j32 && ../../bin/gpu_khop /scratch/bccu/graph/livej/graph 40000 40000
```
