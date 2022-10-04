#include "graph.h"
#include "AvgLinkageUtils/HeapBased.h"
#include "HAC_configuration.h"
#include "HeapBased.h"
#include "NNChainBased.h"

template <class Weights, class Dendrogram>
void WriteDendrogramToDisk(Weights& wgh, Dendrogram& dendrogram,
                           const std::string& of) {
  ofstream out;
  out.open(of);
  size_t wrote = 0;
  for (size_t i = 0; i < dendrogram.size(); i++) {
    if (dendrogram[i].first != i) {
      if (dendrogram[i].first != UINT_E_MAX) {
        out << i << " " << dendrogram[i].first << " "
            << Weights::AsString(dendrogram[i].second) << std::endl;
      }
      wrote++;
    }
  }
  std::cout << "Wrote " << wrote << " parent-pointers. " << std::endl;
}

int main(int argc, char* argv[]) {
  Graph G(argv[1]);
  bool heap_based = true; // P.getOptionValue("-heapbased");
  std::string linkage_opt = "complete"; //P.getOptionValue("-linkage", "complete");01~
  std::cout << "### Application: HAC" << std::endl;
  G.print_meta_data();
  std::cout << "### Params: heap-based = " << heap_based
            << " linkage = " << linkage_opt << std::endl;
  std::cout << "### ------------------------------------" << std::endl;

  timer t;
  t.start();
  double tt;
  std::string of = ""; //P.getOptionValue("-of", "");
  double epsilon = 0.1; //P.getOptionDoubleValue("-epsilon", 0.1);

  if (heap_based) {
    if (linkage_opt == "weightedavg") {
      auto Wghs = WeightedAverageLinkage<Graph, SimilarityClustering, ActualWeight>(G);
      auto dendrogram = heap_based::HAC(G, Wghs);
      tt = t.stop();
      std::cout << "### Running Time: " << tt << std::endl;
      if (!of.empty()) {
        // write merges
        WriteDendrogramToDisk(Wghs, dendrogram, of);
        exit(0);
      }
    } else if (linkage_opt == "complete") {
      auto Wghs = MinLinkage<Graph, SimilarityClustering, ActualWeight>(G);
      auto dendrogram = heap_based::HAC(G, Wghs);
      tt = t.stop();
      std::cout << "### Running Time: " << tt << std::endl;
      if (!of.empty()) {
        // write merges
        WriteDendrogramToDisk(Wghs, dendrogram, of);
        exit(0);
      }
    } else if (linkage_opt == "single") {
      auto Wghs = MaxLinkage<Graph, SimilarityClustering, ActualWeight>(G);
      auto dendrogram = heap_based::HAC(G, Wghs);
      tt = t.stop();
      std::cout << "### Running Time: " << tt << std::endl;
      if (!of.empty()) {
        // write merges
        WriteDendrogramToDisk(Wghs, dendrogram, of);
        exit(0);
      }
    } else if (linkage_opt == "normalizedavg") {
      auto Wghs = NormAverageLinkage<Graph, SimilarityClustering, ActualWeight>(G);
      auto dendrogram = heap_based::HAC(G, Wghs);
      tt = t.stop();
      std::cout << "### Running Time: " << tt << std::endl;
      if (!of.empty()) {
        // write merges
        WriteDendrogramToDisk(Wghs, dendrogram, of);
        exit(0);
      }
    } else if (linkage_opt == "avg") {
      //  if (linkage_opt == "avg") {
      auto Wghs =
          ApproxAverageLinkage<Graph, SimilarityClustering, ActualWeight>(G);
      auto dendrogram = approx_average_linkage::HAC(G, Wghs, epsilon);
      tt = t.stop();
      std::cout << "### Running Time: " << tt << std::endl;
      if (!of.empty()) {
        // write merges
        WriteDendrogramToDisk(Wghs, dendrogram, of);
        exit(0);
      }
    } else {
      std::cerr << "Unknown linkage option: " << linkage_opt << std::endl;
      exit(-1);
    }
  } else {
    if (linkage_opt == "weightedavg") {
      auto Wghs =
          WeightedAverageLinkage<Graph, SimilarityClustering, ActualWeight>(G);
      auto dendrogram = nn_chain::HAC(G, Wghs);
      tt = t.stop();
      std::cout << "### Running Time: " << tt << std::endl;
      if (!of.empty()) {
        // write merges
        WriteDendrogramToDisk(Wghs, dendrogram, of);
        exit(0);
      }
    } else if (linkage_opt == "complete") {
      auto Wghs = MinLinkage<Graph, SimilarityClustering, ActualWeight>(G);
      auto dendrogram = nn_chain::HAC(G, Wghs);
      tt = t.stop();
      std::cout << "### Running Time: " << tt << std::endl;
      if (!of.empty()) {
        // write merges
        WriteDendrogramToDisk(Wghs, dendrogram, of);
        exit(0);
      }
    } else if (linkage_opt == "single") {
      auto Wghs = MaxLinkage<Graph, SimilarityClustering, ActualWeight>(G);
      auto dendrogram = nn_chain::HAC(G, Wghs);
      tt = t.stop();
      std::cout << "### Running Time: " << tt << std::endl;
      if (!of.empty()) {
        // write merges
        WriteDendrogramToDisk(Wghs, dendrogram, of);
        exit(0);
      }
    } else if (linkage_opt == "normalizedavg") {
      auto Wghs =
          NormAverageLinkage<Graph, SimilarityClustering, ActualWeight>(G);
      auto dendrogram = nn_chain::HAC(G, Wghs);
      tt = t.stop();
      std::cout << "### Running Time: " << tt << std::endl;
      if (!of.empty()) {
        // write merges
        WriteDendrogramToDisk(Wghs, dendrogram, of);
        exit(0);
      }
    } else if (linkage_opt == "appx-avg") {
      std::cout << "The approximate average linkage algorithm only supports "
                   "the -heapbased option."
                << std::endl;
      exit(-1);
    } else {
      std::cerr << "Unknown linkage option: " << linkage_opt << std::endl;
      exit(-1);
    }
  }
  return tt;
}
