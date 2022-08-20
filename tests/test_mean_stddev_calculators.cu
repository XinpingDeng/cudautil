#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "cuda_utilities.h"

#include "cpgplot.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <numeric>
#include <vector>
#include <algorithm>
#include <chrono>

#define STRLEN 256

using namespace std;
using namespace doctest;

// Always work with float with test c code
int calculate_mean_stddev(float *data, int ndata, float &mean, float &stddev){

  float mean2 = inner_product(data, data+ndata, data, 0.0)/(float)ndata;
  mean = accumulate(data, data+ndata, 0.0)/(float)ndata;
  stddev = sqrt(mean2 - mean*mean);
    
  return EXIT_SUCCESS;
}

// To check it against wiith given mean and standard deviation is not good
// because random number generator may not generate data strictly follow mean and standard deviation
// for example, because we do not generate enough number of data points

// Here we only check float data type, do we need to check other types?
// We probably should check data type cast directly for other types
TEST_CASE("RealMeanStddevCalculator") {
  
  int ndata = 102400000;
  float mean = 10000;
  float stddev = 1000;
  int nthread = 128;
  float epsilon = 1.0E-4;
  
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
  RealGeneratorNormal normal_data(gen, mean, stddev, ndata);
  print_cuda_memory_info();

  // get mean and stddev with CUDA code
  RealMeanStddevCalculator<float> mean_stddev(normal_data.data, ndata, nthread, 7);
  
  CUDA_STOPTIME(g);
  cout << "elapsed time with GPU is " << gtime << " milliseconds" << endl;

  float mean_g   = mean_stddev.mean;
  float stddev_g = mean_stddev.stddev;
  
  cout << "CUDA mean is " << mean_g
       << " stddev is " << stddev_g
       << endl;
  
  // get mean and stddev from c code
  float mean_c, stddev_c;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  calculate_mean_stddev(normal_data.data, ndata, mean_c, stddev_c);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  cout << "elapsed time with CPU is " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds" << endl;
  
  cout << "C mean is " << mean_c
       << " stddev is " << stddev_c
       << endl;
  
  CHECK(mean_g   == Approx(mean_c).epsilon(epsilon));
  CHECK(stddev_g == Approx(stddev_c).epsilon(epsilon));
}
