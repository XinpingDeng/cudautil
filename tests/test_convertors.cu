#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "util.hpp"
#include "util.cuh"

#include "cpgplot.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#define STRLEN 256

using namespace std;

// We can not compare converted numbers against CPU implementation
// We can only check the difference between original data and converted data with mean and standard deviation
TEST_CASE("RealDataConvertorFloat2Float") {

  int ndata = 102400000;
  float mean = 0;
  float stddev = 10;
  int nthread = 128;

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

  // Convert to float
  RealDataConvertor<float, float> normal_data_float(normal_data.data, ndata, nthread);

  // Get the difference
  RealDataDifferentiator<float, float> normal_data_diff(normal_data.data, normal_data_float.data, ndata, nthread);
  
  // Get mean and standard deviation
  RealDataMeanStddevCalculator<float> mean_stddev(normal_data_diff.data, ndata, nthread, 7);
  
  CUDA_STOPTIME(g);
  cout << "elapsed time is " << gtime << " milliseconds" << endl;

  float mean_f = mean_stddev.mean;
  float stddev_f = mean_stddev.stddev ;
  cout << "mean is " << mean_f << " "
       << "stddev is " << stddev_f 
       << endl;

  // Check numbers
  REQUIRE(mean_f == 0);
  REQUIRE(stddev_f == 0);
}

TEST_CASE("RealDataConvertorFloat2Half") {

  int ndata = 102400000;
  float mean = 0;
  float stddev = 10;
  int nthread = 128;

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

  // Convert to half
  RealDataConvertor<float, half> normal_data_half(normal_data.data, ndata, nthread);

  // Get the difference
  RealDataDifferentiator<float, half> normal_data_diff(normal_data.data, normal_data_half.data, ndata, nthread);
  
  // Get mean and standard deviation
  RealDataMeanStddevCalculator<float> mean_stddev(normal_data_diff.data, ndata, nthread, 7);
  
  CUDA_STOPTIME(g);
  cout << "elapsed time is " << gtime << " milliseconds" << endl;

  float mean_f = mean_stddev.mean;
  float stddev_f = mean_stddev.stddev ;
  cout << "mean is " << mean_f << " "
       << "stddev is " << stddev_f 
       << endl;

  //// Check numbers
  //REQUIRE(mean_f == 0);
  //REQUIRE(stddev_f == 0);
}

TEST_CASE("RealDataConvertorFloat2Int8_T") {

  int ndata = 102400000;
  float mean = 0;
  float stddev = 10;
  int nthread = 128;

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

  // Convert to int8_t
  RealDataConvertor<float, int8_t> normal_data_int8_t(normal_data.data, ndata, nthread);

  // Get the difference
  RealDataDifferentiator<float, int8_t> normal_data_diff(normal_data.data, normal_data_int8_t.data, ndata, nthread);
  
  // Get mean and standard deviation
  RealDataMeanStddevCalculator<float> mean_stddev(normal_data_diff.data, ndata, nthread, 7);
  
  CUDA_STOPTIME(g);
  cout << "elapsed time is " << gtime << " milliseconds" << endl;

  float mean_f = mean_stddev.mean;
  float stddev_f = mean_stddev.stddev ;
  cout << "mean is " << mean_f << " "
       << "stddev is " << stddev_f 
       << endl;

  //// Check numbers
  //REQUIRE(mean_f == 0);
  //REQUIRE(stddev_f == 0);
}
