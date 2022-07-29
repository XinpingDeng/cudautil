#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "util.hpp"
#include "util.cuh"

#include "cpgplot.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <numeric>
#include <vector>
#include <algorithm>

#define STRLEN 256

using namespace std;
using namespace Catch;

// Always work with float with test c code
int calculate_mean_stddev(float *data, int ndata, float &mean, float &stddev){

  vector<float> v(data, data + ndata);
  
  mean = accumulate(v.begin(), v.end(), 0.0)/v.size();
  
  vector<float> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(), [mean](float x) { return x - mean; });
  stddev = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0)/v.size();
  
  //float mean2 =0;
  //
  //mean = 0;
  //for(int i = 0; i < ndata; i++){
  //  float d = data[i];
  //  mean += d;
  //  mean2 += d*d;
  //}  
  //
  //cout << mean << " " << mean2 << endl;
  //
  //mean  /= (float)ndata;
  //mean2 /= (float)ndata;
  //
  //stddev = sqrtf(mean2 - mean*mean);
  //
  //cout << stddev << " " << mean << " " << mean2 << endl;
  
  return EXIT_SUCCESS;
}

// To check it against wiith given mean and standard deviation is not good
// because random number generator may not generate data strictly follow mean and standard deviation
// for example, because we do not generate enough number of data points
TEST_CASE("RealDataMeanStddevCalculator", "RealDataMeanStddevCalculator") {
  
  int ndata = 102400000;
  float mean = 10;
  float stddev = 10;
  int nthread = 128;
  float epsilon = 1.0E-5;
  
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
  
  CUDA_STOPTIME(g);
  cout << "elapsed time is " << gtime << " milliseconds" << endl;

  float mean_g   = mean_stddev.mean;
  float stddev_g = mean_stddev.stddev;
  
  cout << "CUDA mean is " << mean_g
       << " stddev is " << stddev_g
       << endl;
  
  // get mean and stddev from c code
  float mean_c, stddev_c;
  calculate_mean_stddev(normal_data.data, ndata, mean_c, stddev_c);

  cout << "C mean is " << mean_c
       << " stddev is " << stddev_c
       << endl;
  
  REQUIRE(mean_g   == Approx(mean_c).epsilon(epsilon));
  REQUIRE(stddev_g == Approx(stddev_c).epsilon(epsilon));
}
