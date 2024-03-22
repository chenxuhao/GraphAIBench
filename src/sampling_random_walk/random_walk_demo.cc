#include "graph.h"
#include "compressor.hh"
#include "khop.h"

void rWalkSolver(Graph &g, int n_samples, int n_threads)
{
    vector<vidType> inits = get_initial_transits(sample_size(-1) * n_samples, g.V());
    int total_count = (steps() + 1) * n_samples;

    std::vector<vidType> transits(total_count, 0);
    for (int i = 0; i < inits.size(); i++)
    {
        transits[i] = inits[i];
    }
    std::cout << "...initialized starting transits..." << std::endl;

    Timer t;
    t.Start();
    // sample for defined number of steps

    // sampling length is set to `steps()` for all samples
    for (int step = 0; step < steps(); step++)
    {
        std::cout << "STEP " << step << std::endl;

        // sample every new transit in the step for every sample group
        for (int sample_i = 0; sample_i < n_samples; sample_i++)
        {

            vidType sample_transit = transits[step * n_samples + sample_i];

            /// USE THIS?
            // if (old_t == (numeric_limits<uint32_t>::max)())
            // {
            //     transits[t_idx] = (numeric_limits<uint32_t>::max)();
            //     continue;
            // }

            vidType new_t = sample_next_vbyte(g, sample_transit);
            transits[step * (n_samples + 1) + sample_i] = new_t;
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
    std::string out_prefix = argv[2];
    std::string scheme = "streamvbyte";
    bool permutated = false;
    // save_compressed_graph(in_prefix, out_prefix);
    g.load_compressed_graph(out_prefix, scheme, permutated);
    g.print_meta_data();
    std::cout << "LOADED COMPRESSED GRAPH\n"
              << std::endl;

    std::cout << "Begin sampling compressed graph..." << std::endl;
    int n_samples = argc >= 4 ? atoi(argv[3]) : 40000;
    int n_threads = argc >= 5 ? atoi(argv[4]) : 1;
    rWalkSolver(g, n_samples, n_threads);
    return 0;
}
