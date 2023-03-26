#pragma once
#include "unary_compressor.hpp"

class hybrid_compressor : public unary_compressor {
  vidType degree_threashold; // use different compressor for below and above this threashold
}; 
