#pragma once
#include <cstdlib> 
#include <iostream>

const std::string path = std::getenv("DATASET_PATH");

#define NUM_DATASETS 13
const std::string dataset_names[NUM_DATASETS] = {
    "cora", "citeseer", "ppi", "pubmed", "flickr",
    "yelp", "reddit", "amazon", "tester", 
    "ogbn-arxiv", "ogbn-products", "ogbn-proteins", "ogbn-papers100M"};

