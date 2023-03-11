
export GAI_HOME=~/work/GraphAIBench

$GAI_HOME/bin/tc_omp_base ~/datasets/automine/mico/dag 1
$GAI_HOME/bin/tc_omp_vbyte -s streamvbyte ~/datasets/automine/mico/dag-streamvbyte 1
$GAI_HOME/bin/tc_omp_vbyte -s varintgb ~/datasets/automine/mico/dag-varintgb 1

$GAI_HOME/bin/tc_omp_base ~/datasets/automine/livej/dag 1
$GAI_HOME/bin/tc_omp_vbyte -s decomp ~/datasets/automine/livej/dag
$GAI_HOME/bin/tc_omp_vbyte -s cgr ~/datasets/automine/livej/dag-cgr-zeta2
$GAI_HOME/bin/tc_omp_vbyte -s streamvbyte ~/datasets/automine/livej/dag-streamvbyte
$GAI_HOME/bin/tc_omp_vbyte -s varintgb ~/datasets/automine/livej/dag-varintgb

$GAI_HOME/bin/tc_omp_base ~/datasets/automine/orkut/dag 1
$GAI_HOME/bin/tc_omp_vbyte -s streamvbyte ~/datasets/automine/orkut/dag-streamvbyte 1
$GAI_HOME/bin/tc_omp_vbyte -s varintgb ~/datasets/automine/orkut/dag-varintgb 1

$GAI_HOME/bin/tc_omp_cgr-zeta2 ~/datasets/automine/livej/dag-cgr-zeta2
../../bin/tc_omp_vbyte -s s4-bp128-d1 ~/datasets/automine/orkut/dag-s4-bp128-d1

