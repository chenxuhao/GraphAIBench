### Compressed:

```
make rwalk_cpu && ../../bin/rwalk_cpu /home/lgaitskell/data-xhchen/tester/vbyte streamvbyte 30 4000000

make rwalk_cpu && ../../bin/rwalk_cpu /home/lgaitskell/data-xhchen/livej/dag-streamvbyte streamvbyte 30 4000000

make rwalk_omp && ../../bin/rwalk_omp /home/lgaitskell/data-xhchen/livej/dag-streamvbyte streamvbyte 30 4000000 16
```

### Uncompressed:

```
make rwalk_cpu && ../../bin/rwalk_cpu /home/lgaitskell/data-xhchen/livej/graph uncompressed 30 4000000
##  takes 6.86327 sec
make rwalk_omp && ../../bin/rwalk_omp /home/lgaitskell/data-xhchen/livej/graph uncompressed 30 4000000 16
##  takes 8.76526 sec
```

```
interact -n 16 -p parallel -t 2:00:00
cd scratch4/GraphAIBench/src/sampling_random_walk/

make rwalk_omp && /home/lgaitskell/intel/oneapi/vtune/2024.1/bin64/vtune -collect hotspots ../../bin/rwalk_omp /home/lgaitskell/data-xhchen/livej/graph 30 4000000 16
```

Datasets:

```
~/data-xhchen/livej/graph
~/data-xhchen/orkut/graph
~/data-xhchen/twitter40/graph
~/data-xhchen/friendster/graph
~/data-xhchen/uk2007/graph
~/data-xhchen/gsh-2015/graph
~/data-xhchen/clueweb12/graph
~/data-xhchen/uk-2014-csgr/graph
```
