./bin/bfs_omp_base ~/datasets/directed-graphs/web-Google/graph 0

./bin/bfs_omp_direction ~/datasets/directed-graphs/web-Google/graph 0 1

./bin/bfs_gpu_base ~/datasets/directed-graphs/web-Google/graph 0

./bin/bfs_gpu_twc ~/datasets/directed-graphs/web-Google/graph 0

./bin/sssp_omp_base ~/datasets/directed-graphs/web-Google/graph 0 0 1

./bin/sssp_gpu_base ~/datasets/directed-graphs/web-Google/graph 0 0 1

./bin/sssp_gpu_twc ~/datasets/directed-graphs/web-Google/graph 0 0 1

./bin/cc_omp_base ~/datasets/automine/livej/graph

./bin/cc_gpu_base ~/datasets/automine/livej/graph

./bin/cc_gpu_warp ~/datasets/automine/livej/graph

./bin/cc_gpu_afforest ~/datasets/automine/livej/graph

# NOT working yet
./bin/cc_omp_afforest ~/datasets/automine/livej/graph

./bin/bc_omp_base ~/datasets/directed-graphs/web-Google/graph

./bin/pr_omp_base ~/datasets/directed-graphs/soc-LiveJournal1/graph

./bin/pr_omp_push ~/datasets/directed-graphs/soc-LiveJournal1/graph

./bin/cf_omp_base ~/datasets/bipartite-graphs/netflix_mm/graph 0.001 0.00003

./bin/sample_omp_base ../inputs/cora/graph

./bin/tc_omp_base ~/datasets/automine/livej/graph

#./bin/cpu_train_gcn reddit 10 32 softmax

