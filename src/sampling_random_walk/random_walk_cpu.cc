#include "graph.h"
#include "compressor.hh"
#include "khop.h"

void rWalkSolver(Graph &g, bool vbyte, int sample_steps, int n_samples)
{
    vector<vidType> inits = get_initial_transits(sample_size(-1) * n_samples, g.V());
    int total_count = (sample_steps + 1) * n_samples;

    std::vector<vidType> transits(total_count, 0);
    for (int i = 0; i < inits.size(); i++)
    {
        transits[i] = inits[i];
    }
    std::cout << "...initialized starting transits..." << std::endl;

    Timer t;
    t.Start();
    // sample for defined number of steps

    // sampling length is set to `sample_steps` for all samples
    for (int step = 0; step < sample_steps; step++)
    {
        // std::cout << "STEP " << step << std::endl;

        // sample every new transit in the step for every sample group
        for (int sample_i = 0; sample_i < n_samples; sample_i++)
        {

            vidType sample_transit = transits[step * n_samples + sample_i];
            // std::cout << "sample_transit:  at " << step << " " << sample_i << " " << sample_transit << std::endl;
            vidType new_t;
            if (sample_transit == (numeric_limits<uint32_t>::max)())
            {
                new_t = sample_transit;
            }
            else
            {
                if (vbyte)
                {
                    new_t = sample_next_vbyte(g, sample_transit, gen_global);
                }
                else
                {
                    new_t = sample_next(g, sample_transit, gen_global);
                }
            }

            transits[(step + 1) * n_samples + sample_i] = new_t;
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

    std::string scheme = argv[2];
    bool vbyte = (scheme == "streamvbyte");
    if (scheme == "streamvbyte")
    {
        std::string scheme = "streamvbyte";
        bool permutated = false;
        g.load_compressed_graph(in_prefix, scheme, permutated);
        std::cout << "Loaded COMPRESSED Graph\n"
                  << std::endl;
    }
    else if (scheme == "uncompressed")
    {
        g.load_graph(in_prefix);
        std::cout << "Loaded UNcompressed Graph\n"
                  << std::endl;
    }
    else
    {
        std::cout << "Incorrect or no scheme specified\n"
                  << std::endl;
        exit(1);
    }
    g.print_meta_data();

    int sample_steps = atoi(argv[3]);
    int n_samples = argc >= 4 ? atoi(argv[4]) : 40000;
    rWalkSolver(g, vbyte, sample_steps, n_samples);
    return 0;
}
