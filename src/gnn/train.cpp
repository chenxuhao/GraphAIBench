#include "utils.h"
#include "net.h"
std::map<char,double> time_ops;

int main(int argc, char* argv[]) {
  if (argc <= 4 || (argc>9 && argc!=13)) {
    std::cout << "Usage: ./train data num_epochs num_threads type_loss "
              << "hidden(16) score_drop_rate(0.) feat_drop_rate(0.) "
              << "learnng_rate(0.01) num_layers(2) subg_size(0) val_interval(50) inductive(0)\n"
              << "Example: ./bin/cpu_train_gcn citeseer 10 2 softmax\n";
    exit(1);
  }
  #ifdef USE_GAT
  Model<GAT_layer> model;
  std::cout << "Using Graph Attention Network\n";
  #elif USE_SAGE
  Model<SAGE_layer> model;
  std::cout << "Using GraphSAGE\n";
  #else
  Model<GCN_layer> model;
  std::cout << "Using Graph Convolutional Network\n";
  #endif
  model.load_data(argc, argv);
  model.construct_network();
  time_ops[OP_DENSEMM]  = 0.;
  time_ops[OP_SPARSEMM] = 0.;
  time_ops[OP_RELU]     = 0.;
  time_ops[OP_DROPOUT]  = 0.;
  time_ops[OP_LOSS]     = 0.;
  time_ops[OP_BIAS]     = 0.;
  time_ops[OP_REDUCE]   = 0.;
  time_ops[OP_NORM]     = 0.;
  time_ops[OP_SCORE]    = 0.;
  time_ops[OP_ATTN]     = 0.;
  time_ops[OP_TRANSPOSE]= 0.;
  time_ops[OP_SAMPLE]   = 0.;
  time_ops[OP_COPY]     = 0.;
  double t1 = omp_get_wtime();
  model.train();
  double t2 = omp_get_wtime();
  std::cout << "Total training time (validation time included): " << t2-t1 << " seconds\n";
  
  std::cout << "--------------------\n";
  std::cout << "AGGR time: "   << time_ops[OP_SPARSEMM] << "\n";
  std::cout << "LINEAR time: " << time_ops[OP_DENSEMM]  << "\n";
  std::cout << "RELU time: "   << time_ops[OP_RELU]     << "\n";
  std::cout << "DROPOUT time: "<< time_ops[OP_DROPOUT]  << "\n";
  std::cout << "LOSS time: "   << time_ops[OP_LOSS]     << "\n";
  std::cout << "BIAS time: "   << time_ops[OP_BIAS]     << "\n";
  std::cout << "REDUCE time: " << time_ops[OP_REDUCE]   << "\n";
  std::cout << "NORM time: "   << time_ops[OP_NORM]     << "\n";
  std::cout << "SCORE time: "  << time_ops[OP_SCORE]    << "\n";
  std::cout << "ATTN time: "   << time_ops[OP_ATTN]     << "\n";
  std::cout << "TRANSP time: " << time_ops[OP_TRANSPOSE]<< "\n";
  std::cout << "SAMPLE time: " << time_ops[OP_SAMPLE]   << "\n";
  std::cout << "COPY time: "   << time_ops[OP_COPY]     << "\n";
  std::cout << "--------------------\n";
  
  // Testing
  double tt1 = omp_get_wtime();
  auto test_acc = model.evaluate("test");
  double tt2 = omp_get_wtime();
  std::cout << "Test accuracy: " << test_acc << "  test time: " << tt2-tt1 << " seconds\n";
}

