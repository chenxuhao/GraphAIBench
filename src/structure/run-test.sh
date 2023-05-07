
#../../bin/tc_omp_compressed_word-aligned -s hybrid -i ~/data/livej/dag-hybrid-16 -o -p -d 16

../../bin/compressor -s cgr ~/data/orkut/dag ~/data/orkut/dag-zeta2 -z 2
cp ~/data/orkut/dag.meta.txt ~/data/orkut/dag-zeta2.meta.txt
../../bin/tc_gpu_compressed -s cgr -i ~/data/orkut/dag-zeta2 -o

../../bin/compressor -s streamvbyte ~/data/orkut/dag ~/data/orkut/dag-streamvbyte
cp ~/data/orkut/dag.meta.txt ~/data/orkut/dag-streamvbyte.meta.txt
../../bin/tc_gpu_compressed -s streamvbyte -i ~/data/orkut/dag-streamvbyte -o

../../bin/compressor -s hybrid ~/data/orkut/dag ~/data/orkut/dag-hybrid-535 -z 2 -p -d 535 -a 2
cp ~/data/orkut/dag.meta.txt ~/data/orkut/dag-hybrid-535.meta.txt
../../bin/tc_gpu_compressed -s hybrid -i ~/data/orkut/dag-hybrid-535 -o
