#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "util.hpp"
#include "util.cuh"

#include <catch2/catch_test_macros.hpp>

using namespace std;

TEST_CASE("RealDataGeneratorUniform", "RealDataGeneratorUniform") {

  int nthread = 128;
  int ndata = 10240000;
  int exclude = 10;
  int include = 100;
  
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

  RealDataGeneratorUniform uniform_data(gen, ndata, exclude, include, nthread);

  //for(int i = 0; i < ndata; i++){
  //  cout << uniform_data.data[i] << endl;
  //}

  float min = exclude;
  float max = include;
  int nblock = 256;
  RealDataHistogram<float> histogram(uniform_data.data, ndata, min, max, nblock, nthread);

  for(int i = 0; i < NUM_BINS; i++){
    cout << histogram.data[i] << endl;
  }
}    

TEST_CASE("RealDataGeneratorNormal", "RealDataGeneratorNormal") {

  int ndata = 10240000;
  float mean = 0;
  float stddev = 10;
  
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

  RealDataGeneratorNormal normal_data(gen, mean, stddev, ndata);
  print_cuda_memory_info();
  
  //for(int i = 0; i < ndata; i++){
  //  cout << normal_data.data[i] << endl;
  //}
  
  float min = -50;
  float max = 50;
  int nblock = 256;
  int nthread = 128;
  RealDataHistogram<float> histogram(normal_data.data, ndata, min, max, nblock, nthread);

  for(int i = 0; i < NUM_BINS; i++){
    cout << histogram.data[i] << endl;
  }
}
