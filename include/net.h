#pragma once
#include "graph_conv_layer.h"
#include "l2norm_layer.h"
#include "dense_layer.h"
#include "loss_layer.h"
#include "sampler.h"

template <typename gconv_layer>
class Model {
  public:
    Model() { Model(0, 0, 0, 0, 0, 0, 0, 0.0, NULL, "", 0); }
    Model(int epochs, int nl, int nv, int nt, int nc, int di, int dh, float lr, Graph *g, std::string data, bool multi) :
        num_epochs(epochs), num_layers(nl), num_samples(nv), num_threads(nt), num_cls(nc), 
        dim_init(di), dim_hid(dh), lrate(lr), full_graph(g), dataset_name(data), is_sigmoid(multi) {}
    acc_t forward_prop(acc_t& loss);
    acc_t evaluate(std::string type);
    void backward_prop();
    void load_data(int argc, char* argv[]);
    void construct_network();
    void train();
    void update_weights(optimizer* opt);
    void set_netphases(net_phase phase);
    void print_layers_info();
    void transfer_data_to_device();
    // for subgraph sampling
    void subgraph_sampling(int curEpoch, int &num_subg_remain);
    void construct_subg_feats(size_t m, const mask_t* masks);
    void construct_subg_labels(size_t m, const mask_t* masks);
    Graph* get_subg_ptr(int id) { return subgs[id]; }
  private:
    int num_epochs;   // number of epochs
    int num_layers;   // number of graph_conv layers
    int num_samples;  // number of samples, i.e., vertices
    int num_threads;  // number of threads
    int num_cls;      // dimension of initial node features; num of output classes
    int dim_init;     // input feature vector length
    int dim_hid;      // hidden feature vector length
    int subg_size;    // maximum subgraph size when enabling subgraph sampling
    int subg_nv;      // actual subgraph size when enabling subgraph sampling
    int val_interval; // validation performed every interval
    float feat_drop;  // dropout rate for features
    float score_drop; // dropout rate for scores
    float lrate;      // learning rate
    size_t train_begin, train_end, train_count; // vertex id range for training set
    size_t val_begin, val_end, val_count;       // vertex id range for validation set
    size_t test_begin, test_end, test_count;    // vertex id range for testing set
    Graph* full_graph;         // original full graph
    Graph* training_graph;     // training graph: masked by training vertex set
    std::string dataset_name;  // dataset name: citeseer, cora, pubmed, reddit, etc
    bool is_sigmoid;           // single-class (softmax) or multi-class (sigmoid)
    bool use_dense;            // add a Dense layer or not
    bool use_l2norm;           // add a L2norm layer or not
    bool use_gpu;              // use GPU or CPU-only
    bool inductive;            // inductive training or transductive training
    gnn_arch arch;             // GNN architecture: GCN, GAT or SAGE, add selfloop if using GCN
    // layers in the neural network
    std::vector<gconv_layer> layer_gconv;
    loss_layer* layer_loss;
    l2norm_layer* layer_l2norm;
    dense_layer* layer_dense;
    // CPU related
    std::vector<float> input_features;
    std::vector<label_t> labels;
    std::vector<mask_t> masks_train;
    std::vector<mask_t> masks_test;
    std::vector<mask_t> masks_val;
    // GPU related
    float* d_input_features;          // input data on GPU
    label_t* d_labels;
    mask_t* d_masks_train;
    mask_t* d_masks_test;
    mask_t* d_masks_val;

    // for subgraph sampling
    int num_subgraphs;
    Sampler* sampler;
    mask_t* subg_masks;
    vec_t feats_subg;
    std::vector<label_t> labels_subg;
    std::vector<Graph*> subgs; // subgraphs when enabling subgraph sampling
    label_t* d_labels_subg;    // labels for subgraph on device
    float* d_feats_subg;       // input features for subgraph on device
};

