#include "net.h"
#include "reader.h"
#include "math_functions.hh"
#include "softmax_loss_layer.h"
#include "sigmoid_loss_layer.h"
#ifdef ENABLE_GPU
#include "cutils.h"
#endif

template <typename gconv_layer>
void Model<gconv_layer>::load_data(int argc, char* argv[]) {
  dataset_name = std::string(argv[1]);
  num_epochs = atoi(argv[2]);
  num_threads = atoi(argv[3]);
  omp_set_num_threads(num_threads);
#ifdef USE_MKL
  mkl_set_num_threads(num_threads);
#endif
  is_sigmoid = std::string(argv[4])=="sigmoid" ? true:false;
  use_dense = false;
  use_l2norm = false;
  arch = gnn_arch::GCN;
#ifdef USE_GAT
  arch = gnn_arch::GAT;
#elif USE_SAGE
  arch = gnn_arch::SAGE;
#endif

  feat_drop = 0.;
  score_drop = 0.;
  inductive = false;
  dim_hid = DEFAULT_SIZE_HID;
  lrate = DEFAULT_RATE_LEARN;
  num_layers = DEFAULT_NUM_LAYER;
  subg_size = 0;
  val_interval = EVAL_INTERVAL;
  if (argc == 6) {
    dim_hid = atoi(argv[5]);
  } else if (argc == 7) {
    dim_hid = atoi(argv[5]);
    score_drop = atof(argv[6]);
  } else if (argc == 8) {
    dim_hid = atoi(argv[5]);
    score_drop = atof(argv[6]);
    feat_drop = atof(argv[7]);
  } else if (argc == 9) {
    dim_hid = atoi(argv[5]);
    score_drop = atof(argv[6]);
    feat_drop = atof(argv[7]);
    lrate = atof(argv[8]);
  } else if (argc > 9) {
    assert(argc == 13);
    dim_hid = atoi(argv[5]);
    score_drop = atof(argv[6]);
    feat_drop = atof(argv[7]);
    lrate = atof(argv[8]);
    num_layers = atoi(argv[9]);
    subg_size = atoi(argv[10]);
    val_interval = atoi(argv[11]);
    inductive = atoi(argv[12]);
  }
  assert(num_layers >= 2);

  // l2norm+dense layer is useful for sampling and GAT
  if (subg_size > 0 || arch == gnn_arch::GAT) use_l2norm = true;
  if (use_l2norm) use_dense = true;

  use_gpu = false;
#ifdef ENABLE_GPU
  use_gpu = true;
#endif
  full_graph = new Graph(use_gpu);
  auto reader = new Reader(dataset_name);
  reader->readGraphFromGRFile(full_graph);
  num_samples = full_graph->size();
  //full_graph->print_graph();
  if (arch != gnn_arch::SAGE)
    full_graph->add_selfloop();
  dim_init = reader->read_features(input_features);
  num_cls = reader->read_labels(labels, !is_sigmoid);
#ifdef CSR_SEGMENTING
  full_graph->segmenting(dim_hid);
#endif
  if (subg_size > 0 && val_interval < num_epochs && use_gpu) {
    std::cout << "disabling validation for subgraph sampling on GPU\n";
    val_interval = num_epochs;
  }
  std::cout << "num_threads = " << num_threads
            << ", num_vertices = " << num_samples
            << ", num_edges = " << full_graph->sizeEdges()
            << ", num_layers = " << num_layers
            << ", \nnum_epochs = " << num_epochs
            << ", input_length = " << dim_init
            << ", hidden_length = " << dim_hid
            << ", num_classes = " << num_cls 
            << ", \nfeat_drop = " << feat_drop
            << ", score_drop = " << score_drop
            << ", subg_size = " << subg_size
            << ", val_interval = " << val_interval
            << ", learning_rate = " << lrate << "\n";
  masks_train.resize(num_samples);
  masks_test.resize(num_samples);
  masks_val.resize(num_samples);
  if (dataset_name == "reddit") {
    train_begin = 0;
    train_count = 153431;
    train_end   = train_begin + train_count;
    val_begin   = 153431;
    val_count   = 23831;
    val_end     = val_begin + val_count;
    #pragma omp parallel for
    for (size_t i = train_begin; i < train_end; i++)
      masks_train[i] = 1;
    #pragma omp parallel for
    for (size_t i = val_begin; i < val_end; i++)
      masks_val[i] = 1;
    test_begin = 177262;
    test_count = 55703;
    test_end = test_begin + test_count;
    #pragma omp parallel for
    for (size_t i = test_begin; i < test_end; i++)
      masks_test[i] = 1;
    std::cout << "train_mask range: [" << train_begin << ", "
            << train_end << ") Number of valid samples: " << train_count << " ("
            << (float)train_count / (float)num_samples * (float)100 << "\%)\n";
    std::cout << "val_mask range: [" << val_begin << ", "
            << val_end << ") Number of valid samples: " << val_count << " ("
            << (float)val_count / (float)num_samples * (float)100 << "\%)\n";
    std::cout << "test_mask range: [" << test_begin << ", "
            << test_end << ") Number of valid samples: " << test_count << " ("
            << (float)test_count / (float)num_samples * (float)100 << "\%)\n";
  } else {
    train_count = reader->read_masks("train", num_samples, train_begin, train_end, masks_train.data());
    val_count = reader->read_masks("val", num_samples, val_begin, val_end, masks_val.data());
    test_count = reader->read_masks("test", num_samples, test_begin, test_end, masks_test.data());
  }
  // for sampling
  assert(size_t(subg_size) <= train_count);
  num_subgraphs = num_threads;
  if (subg_size > 0) inductive = true;
  if (inductive) training_graph = full_graph->generate_masked_graph(&masks_train[0]);
  else training_graph = full_graph;
  if (subg_size > 0) {
    assert(inductive);
    //std::cout << "Allocating sampler\n";
    sampler = new Sampler(full_graph, training_graph, &masks_train[0], train_count);
    subgs.resize(num_subgraphs);
    //std::cout << "Allocating sampled subgraphs\n";
    for (int i = 0; i < num_subgraphs; i++) {
      subgs[i] = new Graph(use_gpu);
      subgs[i]->alloc_on_device(subg_size);
    }
    subg_masks = new mask_t[num_samples * num_subgraphs];
  }
  if (use_gpu) {
    if (subg_size > 0) {
      float_malloc_device(subg_size*dim_init, d_feats_subg);
      size_t labels_size = subg_size;
      if (is_sigmoid) labels_size *= num_cls;
      uint8_malloc_device(labels_size, d_labels_subg);
    } else {
      transfer_data_to_device();
      training_graph->alloc_on_device();
      training_graph->copy_to_gpu();
    }
  }
  if (subg_size == 0 || !use_gpu) {
#if (defined(USE_MKL) && defined(PRECOMPUTE_SCORES)) || (defined(ENABLE_GPU) && defined(USE_CUSPARSE))
    if (inductive) training_graph->compute_edge_data();
    else full_graph->compute_edge_data();
#else
    if (inductive) training_graph->compute_vertex_data();
    else full_graph->compute_vertex_data();
#endif
  }
}

template <typename gconv_layer>
void Model<gconv_layer>::transfer_data_to_device() {
  //std::cout << "transfer features to GPU\n";
  float_malloc_device(num_samples*dim_init, d_input_features);
  copy_float_device(num_samples*dim_init, &input_features[0], d_input_features);
  //std::cout << "transfer labels to GPU\n";
  if (is_sigmoid) {
    uint8_malloc_device(num_samples*num_cls, d_labels);
    copy_uint8_device(num_samples*num_cls, &labels[0], d_labels);
  } else {
    uint8_malloc_device(num_samples, d_labels);
    copy_uint8_device(num_samples, &labels[0], d_labels);
  }
  //std::cout << "transfer masks to GPU\n";
  uint8_malloc_device(num_samples, d_masks_train);
  copy_uint8_device(num_samples, &masks_train[0], d_masks_train);
  uint8_malloc_device(num_samples, d_masks_test);
  copy_uint8_device(num_samples, &masks_test[0], d_masks_test);
  uint8_malloc_device(num_samples, d_masks_val);
  copy_uint8_device(num_samples, &masks_val[0], d_masks_val);
}

template <typename gconv_layer>
void Model<gconv_layer>::update_weights(optimizer* opt) {
  //std::cout << "Updating weights\n";
  //regularize();
  for (int i = 0; i < num_layers; i++)
    layer_gconv[i].update_weight(opt);
}

//! set netphases for all layers in this network
template <typename gconv_layer>
void Model<gconv_layer>::set_netphases(net_phase phase) {
  for (int i = 0; i < num_layers; i++)
    layer_gconv[i].set_netphase(phase);
  layer_loss->set_netphase(phase);
}

//! print all layers
template <typename gconv_layer>
void Model<gconv_layer>::print_layers_info() {
  for (int i = 0; i < num_layers; i++)
    layer_gconv[i].print_layer_info();
  layer_loss->print_layer_info();
}


template <typename gconv_layer>
void Model<gconv_layer>::construct_subg_feats(size_t m, const mask_t* masks) {
  size_t count = 0;
  feats_subg.resize(m * dim_init);
  for (int i = 0; i < num_samples; i++) {
    if (masks[i] == 1) {
      std::copy(&input_features[size_t(i) * size_t(dim_init)], 
                &input_features[size_t(i+1) * size_t(dim_init)],
                &feats_subg[count * dim_init]);
      count++;
    }
  }
  assert(count == m);
}

template <typename gconv_layer>
void Model<gconv_layer>::construct_subg_labels(size_t m, const mask_t* masks) {
  if (is_sigmoid) labels_subg.resize(m * num_cls);
  else labels_subg.resize(m);
  size_t count = 0;
  // see which labels to copy over for this subgraph
  for (int i = 0; i < num_samples; i++) {
    if (masks[i] == 1) {
      if (is_sigmoid)
        std::copy(&labels[i * num_cls], &labels[(i + 1) * num_cls],
                  &labels_subg[count * num_cls]);
      else labels_subg[count] = labels[i];
      count++;
    }
  }
  assert(count == m); 
}

// subgraph sampling
template <typename gconv_layer>
void Model<gconv_layer>::subgraph_sampling(int curEpoch, int &num_subg_remain) {
  if (num_subg_remain == 0) {
    //std::cout << "Generating " << num_subgraphs << " subgraph(s)\n";
    double t1 = omp_get_wtime();
    #pragma omp parallel for
    for (int sid = 0; sid < num_subgraphs; sid++) {
      VertexSet sampledSet;
      auto tid = omp_get_thread_num();
      auto seed = tid;
      //auto seed = curEpoch;
      auto nv = sampler->select_vertices(subg_size, sampledSet, seed);
      //std::cout << nv << " vertices seleceted\n";
      sampler->generateSubgraph(sampledSet, &subg_masks[sid*num_samples], get_subg_ptr(sid));
    }
    num_subg_remain = num_subgraphs;
    double t2 = omp_get_wtime();
    time_ops[OP_SAMPLE] += t2 - t1;
  }
  double t3 = omp_get_wtime();
  // count their degrees
  for (int i = 0; i < num_subgraphs; i++) {
    auto sg_ptr = get_subg_ptr(i);
    sg_ptr->degree_counting();
  }
  // choose a subgraph to use
  num_subg_remain--;
  int sg_id     = num_subg_remain;
  auto subg_ptr = get_subg_ptr(sg_id);
  subg_nv = subg_ptr->size();

#ifdef ENABLE_GPU
  if (arch != gnn_arch::SAGE) subg_ptr->add_selfloop();
  subg_ptr->alloc_on_device();
  subg_ptr->copy_to_gpu();
#endif
  // update normalization scores for GCN
#if (defined(USE_MKL) && defined(PRECOMPUTE_SCORES)) || (defined(ENABLE_GPU) && defined(USE_CUSPARSE))
  subg_ptr->compute_edge_data();
#else
  subg_ptr->compute_vertex_data();
#endif
  for (int i = 0; i < num_layers; i++) {
    layer_gconv[i].update_dim_size(subg_nv);
    layer_gconv[i].set_graph_ptr(subg_ptr);
  }
  if (use_l2norm) layer_l2norm->update_dim_size(subg_nv);
  if (use_dense) layer_dense->update_dim_size(subg_nv);
  // update labels for the loss layer
  construct_subg_labels(subg_nv, &subg_masks[sg_id * num_samples]);
  layer_loss->update_dim_size(subg_nv);
  // update features for subgraph
  construct_subg_feats(subg_nv, &subg_masks[sg_id * num_samples]);
#ifdef ENABLE_GPU
  //copy_float_device(subg_nv*dim_init, &feats_subg[0], d_feats_subg);
  copy_async_device<float>(subg_nv*dim_init, &feats_subg[0], d_feats_subg);
  layer_gconv[0].set_feat_in(d_feats_subg); // feed input datd
  size_t labels_size = subg_nv;
  if (is_sigmoid) labels_size *= num_cls;
  //copy_uint8_device(labels_size, &labels_subg[0], d_labels_subg);
  copy_async_device<label_t>(labels_size, &labels_subg[0], d_labels_subg);
  layer_loss->set_labels_ptr(d_labels_subg);
#else
  layer_gconv[0].set_feat_in(&feats_subg[0]); // feed input data
  layer_loss->set_labels_ptr(&labels_subg[0]);
#endif
  double t4 = omp_get_wtime();
  time_ops[OP_COPY] += t4 - t3;
  //std::cout << "Sampling done!\n";
}

template <typename gconv_layer>
void Model<gconv_layer>::train() {
  optimizer* opt = new adam(lrate);
  std::cout << "Start training...\n";
  double total_train_time = 0.0;
  int num_subg_remain = 0;

  for (int itr = 0; itr < num_epochs; itr ++) {
    if (subg_size > 0) subgraph_sampling(itr, num_subg_remain);
    std::cout << "Epoch " << std::setw(3) << itr << " ";
    // training
    set_netphases(net_phase::TRAIN);
    acc_t train_loss = 0.0;
    double t_f1 = omp_get_wtime();
    auto train_acc = forward_prop(train_loss);
    double t_f2 = omp_get_wtime();
    double fw_time = t_f2 - t_f1;
    double t_b1 = omp_get_wtime();
    backward_prop();
    update_weights(opt);
    double t_b2 = omp_get_wtime();
    double bw_time = t_b2 - t_b1;
    double epoch_time = fw_time + bw_time;
    total_train_time += epoch_time;
    std::cout << "train_loss " << std::setprecision(3) << std::fixed
              << train_loss << " train_acc " << train_acc << " ";

    // validation
    if (itr%val_interval== 0 && itr != 0) {
      double t_v1 = omp_get_wtime();
      auto val_acc = evaluate("val");
      double t_v2 = omp_get_wtime();
      double val_time = t_v2 - t_v1;
      std::cout << "val_acc " << std::setprecision(3) << std::fixed
                << val_acc << " ";
      std::cout << "time " << std::setprecision(3) << std::fixed
                << epoch_time + val_time << " s (train_time "
                << epoch_time << " val_time " << val_time << ")\n";
      if (inductive) { // back to training
        for (int i = 0; i < num_layers; i++)
          layer_gconv[i].set_graph_ptr(training_graph);
      }
    } else {
      std::cout << "train_time " << std::fixed << epoch_time
                << " s (fw " << fw_time << ", bw " << bw_time << ")\n";
    }
  }
  double avg_train_time = total_train_time / (double)num_epochs;
  double throughput     = (double)num_epochs / total_train_time;
  std::cout << "Average training time per epoch: " << avg_train_time
            << " seconds. Throughput " << throughput << " epoch/s\n";
  if (subg_size > 0) {
    for (int i = 0; i < num_subgraphs; i++)
      subgs[i]->dealloc();
#ifdef ENABLE_GPU
    free_device<float>(d_feats_subg);
    free_device<label_t>(d_labels_subg);
#endif
  }
  if (inductive) training_graph->dealloc();
}

template <typename gconv_layer>
void Model<gconv_layer>::construct_network() {
  std::cout << "constructing neural network...\n";
  auto nv = num_samples;
  if (subg_size > 0 && use_gpu) nv = subg_size; // save memory for GPU
  for (int l = 0; l < num_layers-1; l++) {
    int dim_in = (l == 0) ? dim_init : dim_hid;
    layer_gconv.push_back(gconv_layer(l, nv, dim_in, dim_hid, training_graph, true, lrate, feat_drop, score_drop));
  }
  int dim_out = num_cls;
  if (use_dense) dim_out = dim_hid;
  layer_gconv.push_back(gconv_layer(num_layers-1, nv, dim_hid, dim_out, training_graph, false, lrate, feat_drop, score_drop));
  if (use_l2norm) layer_l2norm = new l2norm_layer(nv, dim_hid);
  if (use_dense) layer_dense = new dense_layer(nv, dim_hid, num_cls, lrate);
#ifdef ENABLE_GPU
    layer_gconv[0].set_feat_in(d_input_features);
#else
    layer_gconv[0].set_feat_in(&input_features[0]);
#endif
  label_t* labels_ptr = &labels[0];
#ifdef ENABLE_GPU
  labels_ptr = d_labels;
#endif
  if (is_sigmoid) {
    layer_loss = new sigmoid_loss_layer(nv, num_cls, labels_ptr);
  } else {
    layer_loss = new softmax_loss_layer(nv, num_cls, labels_ptr);
  }
  //print_layers_info();
}

// forward pass in training phase
template <typename gconv_layer>
acc_t Model<gconv_layer>::forward_prop(acc_t& loss) {
  for (int l = 0; l < num_layers-1; l++)
    layer_gconv[l].forward(layer_gconv[l+1].get_feat_in());
  if (use_dense) {
    if (use_l2norm) {
      layer_gconv[num_layers-1].forward(layer_l2norm->get_feat_in());
      layer_l2norm->forward(layer_dense->get_feat_in());
    } else 
      layer_gconv[num_layers-1].forward(layer_dense->get_feat_in());
    layer_dense->forward(layer_loss->get_feat_in());
  } else {
    layer_gconv[num_layers-1].forward(layer_loss->get_feat_in());
  }

  mask_t* masks_ptr = &masks_train[0];
  label_t* labels_ptr = &labels[0];
#ifdef ENABLE_GPU
  masks_ptr = d_masks_train;
  labels_ptr = d_labels;
#endif
  size_t begin = train_begin, end = train_end, count = train_count;
  if (subg_size > 0) {
#ifdef ENABLE_GPU
    labels_ptr = d_labels_subg;
#else
    labels_ptr = &labels_subg[0];
#endif
    masks_ptr = NULL;
    begin = 0;
    end = subg_nv;
    count = subg_nv;
  }
  layer_loss->forward(begin, end, masks_ptr);
  loss = layer_loss->get_prediction_loss(begin, end, count, masks_ptr);
  // prediction error
  // Squared Norm Regularization to mitigate overfitting
  //loss += weight_decay * layers[0]->get_weight_decay_loss();
  acc_t accuracy = 0.0;
  if (is_sigmoid)
    accuracy = masked_accuracy_multi(begin, end, count, num_cls, masks_ptr, layer_loss->get_feat_out(), labels_ptr);
  else
    accuracy = masked_accuracy_single(begin, end, count, num_cls, masks_ptr, layer_loss->get_feat_in(), labels_ptr);
  return accuracy;
}

// forward pass in testing or invalidation phase
template <typename gconv_layer>
acc_t Model<gconv_layer>::evaluate(std::string type) {
  set_netphases(net_phase::TEST);
  if (subg_size > 0 || inductive) {
    // not training; switch back to use the full graph
    for (int i = 0; i < num_layers; i++)
      layer_gconv[i].set_graph_ptr(full_graph);
#ifdef ENABLE_GPU
    full_graph->alloc_on_device();
    full_graph->copy_to_gpu();
  #ifdef USE_CUSPARSE
    full_graph->compute_edge_data();
  #else
    full_graph->compute_vertex_data();
  #endif
#endif
    // for sampling, also need to switch back the input features and labels
    if (subg_size > 0) {
      for (int i = 0; i < num_layers; i++)
        layer_gconv[i].update_dim_size(num_samples);
      if (use_dense) layer_dense->update_dim_size(num_samples);
      if (use_l2norm) layer_l2norm->update_dim_size(num_samples);
      layer_loss->update_dim_size(num_samples);
#ifdef ENABLE_GPU
      transfer_data_to_device();
      layer_gconv[0].set_feat_in(d_input_features);
      layer_loss->set_labels_ptr(d_labels);
#else
      layer_gconv[0].set_feat_in(&input_features[0]);
      layer_loss->set_labels_ptr(&labels[0]);
#endif
    }
  }
  for (int l = 0; l < num_layers-1; l++)
    layer_gconv[l].forward(layer_gconv[l+1].get_feat_in());
  if (use_dense) {
    if (use_l2norm) {
      layer_gconv[num_layers-1].forward(layer_l2norm->get_feat_in());
      layer_l2norm->forward(layer_dense->get_feat_in());
    } else 
      layer_gconv[num_layers-1].forward(layer_dense->get_feat_in());
    layer_dense->forward(layer_loss->get_feat_in());
  } else {
    layer_gconv[num_layers-1].forward(layer_loss->get_feat_in());
  }
 
  mask_t* masks_ptr = &masks_val[0];
  label_t* labels_ptr = &labels[0];
#ifdef ENABLE_GPU
  masks_ptr = d_masks_val;
  labels_ptr = d_labels;
#endif
  size_t begin = val_begin, end = val_end, count = val_count;
  if (type == "test") {
#ifdef ENABLE_GPU
    masks_ptr = d_masks_test;
#else
    masks_ptr = &masks_test[0];
#endif
    begin = test_begin;
    end = test_end;
    count = test_count;
  }
  acc_t accuracy = 0.0;
  if (is_sigmoid) {
    layer_loss->forward(begin, end, masks_ptr);
    accuracy = masked_accuracy_multi(begin, end, count, num_cls, masks_ptr, layer_loss->get_feat_out(), labels_ptr);
  } else
    accuracy = masked_accuracy_single(begin, end, count, num_cls, masks_ptr, layer_loss->get_feat_in(), labels_ptr);
  return accuracy;
}

template <typename gconv_layer>
void Model<gconv_layer>::backward_prop() {
  mask_t* masks_ptr = &masks_train[0];
#ifdef ENABLE_GPU
  masks_ptr = d_masks_train;
#endif
  size_t begin = train_begin, end = train_end;
  if (subg_size > 0) {
    masks_ptr = NULL;
    begin = 0;
    end = subg_nv;
  }
  if (use_dense) {
    layer_loss->backward(begin, end, masks_ptr, layer_dense->get_grad_in());
    if (use_l2norm) {
      layer_dense->backward(layer_l2norm->get_grad_in());
      layer_l2norm->backward(layer_gconv[num_layers-1].get_grad_in());
      layer_gconv[num_layers-1].backward(layer_l2norm->get_feat_in(), layer_gconv[num_layers-2].get_grad_in());
    } else {
      layer_dense->backward(layer_gconv[num_layers-1].get_grad_in());
      layer_gconv[num_layers-1].backward(layer_dense->get_feat_in(), layer_gconv[num_layers-2].get_grad_in());
    }
  } else {
    layer_loss->backward(begin, end, masks_ptr, layer_gconv[num_layers-1].get_grad_in());
    layer_gconv[num_layers-1].backward(layer_loss->get_feat_in(), layer_gconv[num_layers-2].get_grad_in());
  }
  for (int l = num_layers-2; l > 0; l--)
    layer_gconv[l].backward(layer_gconv[l+1].get_feat_in(), layer_gconv[l-1].get_grad_in());
  layer_gconv[0].backward(layer_gconv[1].get_feat_in(), NULL);
}

template class Model<GCN_layer>;
template class Model<GAT_layer>;
template class Model<SAGE_layer>;
