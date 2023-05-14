#pragma once

const std::string path =
    // "/jet/home/xhchen/datasets/Learning/"; // path to the input dataset
    "/home/adrian/Documents/git/GraphAIBench/inputs/"; // path to the input dataset
//    "/h2/xchen/datasets/Learning/"; // path to the input dataset
//    "/ocean/projects/cie170003p/shared/Learning/"; // path to the input dataset

#define NUM_DATASETS 13
const std::string dataset_names[NUM_DATASETS] = {
    "cora", "citeseer", "ppi", "pubmed", "flickr",
    "yelp", "reddit", "amazon", "tester", 
    "ogbn-arxiv", "ogbn-products", "ogbn-proteins", "ogbn-papers100M"};

