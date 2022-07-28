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
  int ndata = 10240;
  int exclude = 10;
  int include = 100;
  
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

  RealDataGeneratorUniform uniform_data(gen, ndata, exclude, include, nthread);

  for(int i = 0; i < ndata; i++){
    cout << uniform_data.data[i] << endl;
  }
}    
