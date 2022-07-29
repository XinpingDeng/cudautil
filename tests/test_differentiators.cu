#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "util.hpp"
#include "util.cuh"

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

// Here we only check float data type, do we need to check other types?
// We probably should check data type cast directly for other types
TEST_CASE("RealDataDifferentiator") {
  
  int ndata = 102400000;
  float mean = 10;
  float stddev = 10;
  int nthread = 128;
  float epsilon = 1.0E-6;
  
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
  RealDataGeneratorNormal normal_data1(gen, mean, stddev, ndata);
  RealDataGeneratorNormal normal_data2(gen, mean, stddev, ndata);
  print_cuda_memory_info();

  // get difference with CUDA code
  RealDataDifferentiator<float, float> diff_g(normal_data1.data, normal_data2.data, ndata, nthread);

  // Get mean and stddev with CUDA code
  RealDataMeanStddevCalculator<float> mean_stddev_g(diff_g.data, ndata, nthread, 7);

  CUDA_STOPTIME(g);
  cout << "elapsed time with GPU is " << gtime << " milliseconds" << endl;

  float mean_g   = mean_stddev_g.mean;
  float stddev_g = mean_stddev_g.stddev;
  
  cout << "GPU mean is " << mean_g
       << " stddev is " << stddev_g
       << endl;

  // Work on CPU
  ManagedMemoryAllocator<float> diff_c(ndata);
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();

  // Get difference with CPU
  transform(normal_data1.data, normal_data1.data+ndata, normal_data2.data, diff_c.data, minus<float>());

  // Get mean and stddev with CUDA code
  // I only care about difference, so I use cuda mean and standard deviation calculator
  RealDataMeanStddevCalculator<float> mean_stddev_c(diff_c.data, ndata, nthread, 7);

  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  cout << "elapsed time with CPU is " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " milliseconds" << endl;
  
  float mean_c   = mean_stddev_c.mean;
  float stddev_c = mean_stddev_c.stddev;
  
  cout << "CPU mean is " << mean_c
       << " stddev is " << stddev_c
       << endl;

  CHECK(mean_g   == Approx(mean_c).epsilon(epsilon));
  CHECK(stddev_g == Approx(stddev_c).epsilon(epsilon));
}
