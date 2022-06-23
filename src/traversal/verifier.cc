// Copyright 2020 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"

void BFSVerifier(Graph &g, int source, vidType *depth_to_test) {
	std::cout << "Verifying...\n";
  auto m = g.V();
	vector<vidType> depth(m, MYINFINITY);
	vector<int> to_visit;
	Timer t;
	t.Start();
	depth[source] = 0;
	to_visit.reserve(m);
	to_visit.push_back(source);
	for (std::vector<int>::iterator it = to_visit.begin(); it != to_visit.end(); it++) {
		int src = *it;
    for (auto dst : g.N(src)) {
			if (depth[dst] == MYINFINITY) {
				depth[dst] = depth[src] + 1;
				to_visit.push_back(dst);
			}
		}
	}
	t.Stop();
	std::cout << "\truntime [serial] = " << t.Seconds() << " sec\n";

	// Report any mismatches
	bool all_ok = true;
	for (int n = 0; n < m; n ++) {
		if (depth_to_test[n] != depth[n]) {
			//std::cout << n << ": " << depth_to_test[n] << " != " << depth[n] << std::endl;
			all_ok = false;
		}
	}
	if(all_ok) std::cout << "Correct\n";
	else std::cout << "Wrong\n";
}

