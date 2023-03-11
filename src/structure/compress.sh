
#../../bin/test_cgr_compressor ~/datasets/automine/citeseer/dag ~/datasets/automine/citeseer/dag-cgr
../../bin/vbyte_compressor -s streamvbyte ~/datasets/automine/livej/dag ~/datasets/automine/livej/dag-streamvbyte

../../bin/vbyte_compressor -s varintgb ~/datasets/automine/livej/dag ~/datasets/automine/livej/dag-varintgb
cp ~/datasets/automine/livej/dag.meta.txt ~/datasets/automine/livej/dag-varintgb.meta.txt

../../bin/vbyte_compressor -s varintgb ~/datasets/automine/orkut/dag ~/datasets/automine/orkut/dag-varintgb
cp ~/datasets/automine/orkut/dag.meta.txt ~/datasets/automine/orkut/dag-varintgb.meta.txt

