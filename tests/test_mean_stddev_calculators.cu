#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "util.hpp"
#include "util.cuh"

#include "cpgplot.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#define STRLEN 256

using namespace std;
using namespace Catch;

// Always work with float with test c code
int calculate_mean(float *data, int ndata, float &mean, float &mean2){

  mean = 0;
  mean2 =0;
  
  for(int i = 0; i < ndata; i++){
    float d = data[i];
    mean += d;
    mean2 += d*d;
  }  

  mean  /= (float)ndata;
  mean2 /= (float)ndata;
  
  return EXIT_SUCCESS;
}

int calculate_stddev(float mean, float mean2, float &stddev){

  stddev = sqrtf(mean2 - mean*mean);
  
  return EXIT_SUCCESS;
}

TEST_CASE("RealDataMeanStddevCalculator", "RealDataMeanStddevCalculator") {

  int ndata = 102400000;
  float mean = 0;
  float stddev = 10;
  int nthread = 128;
  float epsilon = 1.0E-3;
  
  cudaEvent_t g_start;
  cudaEvent_t g_stop;
  float gtime = 0;
  checkCudaErrors(cudaEventCreate(&g_start));
  checkCudaErrors(cudaEventCreate(&g_stop));
  CUDA_STARTTIME(g);
  
  // Get float data
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
  RealDataGeneratorNormal normal_data(gen, mean, stddev, ndata);
  print_cuda_memory_info();

  // get mean and stddev with CUDA code
  RealDataMeanStddevCalculator<float> mean_stddev(normal_data.data, ndata, nthread, 7);

  cout << "CUDA mean is " << mean_stddev.mean
       << " stddev is " << mean_stddev.stddev
       << endl;
  
  CUDA_STOPTIME(g);
  cout << "elapsed time is " << gtime << " milliseconds" << endl;

  // get mean and stddev from c code
  float mean_c, mean2, stddev_c;
  calculate_mean(normal_data.data, ndata, mean_c, mean2);
  calculate_stddev(mean_c, mean2, stddev_c);

  cout << "C mean is " << mean_c
       << " stddev is " << stddev
       << endl;
  
  REQUIRE(mean_stddev.mean   == Approx(mean_c).epsilon(epsilon));
  REQUIRE(mean_stddev.stddev == Approx(stddev).epsilon(epsilon));
}
