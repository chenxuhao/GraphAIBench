#include <omp.h>
#include "graph.h"
#include "compressor.hh"
#include "khop.h"

void rWalkOMPSolver(Graph &g, int sample_steps, int n_samples, int n_threads)
{
    int num_threads = 1;
    omp_set_num_threads(n_threads);
#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    vector<vidType> inits = get_initial_transits(sample_size(-1) * n_samples, g.V());
    int total_count = (sample_steps + 1) * n_samples;

    std::vector<vidType> transits(total_count, 0);
#pragma omp parallel for
    for (int i = 0; i < inits.size(); i++)
    {
        transits[i] = inits[i];
    }
    std::cout << "...initialized starting transits..." << std::endl;

    Timer t;
    t.Start();
    // sample for defined number of steps

    // sampling length is set to `sample_steps` for all samples

#pragma omp parallel
    {
        int t_idx = omp_get_thread_num();
        std::mt19937 gen(t_idx);

        // sample every new transit in the step for every sample group
#pragma omp for // schedule(dynamic) // schedule(static) num_threads(8)
        for (int sample_i = 0; sample_i < n_samples; sample_i++)
        {
            for (int step = 0; step < sample_steps; step++)
            {
                // std::cout << "STEP " << step << std::endl;

                vidType sample_transit = transits[step * n_samples + sample_i];
                // std::cout << "sample_transit:  at " << step << " " << sample_i << " " << sample_transit << std::endl;

                vidType new_t;
                if (sample_transit == (numeric_limits<uint32_t>::max)())
                {
                    new_t = sample_transit;
                }
                else
                {
                    new_t = sample_next_vbyte(g, sample_transit, gen);
                }

                transits[(step + 1) * n_samples + sample_i] = new_t;
            }
        }
    }
    t.Stop();
    std::cout << "result size: " << total_count << std::endl;
    std::cout << "Finished sampling in " << t.Seconds() << " sec" << std::endl;
}

int main(int argc, char *argv[])
{
    Graph g;
    std::string in_prefix = argv[1];
    std::string scheme = "streamvbyte";
    bool permutated = false;
    g.load_graph(in_prefix);
    g.load_compressed_graph(in_prefix, scheme, permutated);
    g.print_meta_data();
    std::cout << "LOADED COMPRESSED GRAPH\n"
              << std::endl;

    int sample_steps = atoi(argv[2]);
    int n_samples = argc >= 4 ? atoi(argv[3]) : 40000;
    int n_threads = argc >= 5 ? atoi(argv[4]) : 1;
    std::cout << "Begin OpenMP sampling compressed graph..." << std::endl;
    rWalkOMPSolver(g, sample_steps, n_samples, n_threads);
    return 0;
}
