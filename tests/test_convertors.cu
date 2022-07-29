#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "util.hpp"
#include "util.cuh"

#include "cpgplot.h"

#include <catch2/catch_test_macros.hpp>

#define STRLEN 256

using namespace std;

TEST_CASE("RealDataConvertor", "RealDataConvertor") {

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
  cout << "normal data mean is " << mean_stddev.mean << "\t"
       << "normal data stddev is " << mean_stddev.stddev 
       << endl;
  
  //// Get histogram
  //float min = -50;
  //float max = 50;
  //int nblock = 256;
  //RealDataHistogram<float> histogram(normal_data.data, ndata, min, max, nblock, nthread);
  //
  CUDA_STOPTIME(g);
  cout << "elapsed time is " << gtime << " milliseconds" << endl;
  
  //// plot histogram
  //char device[STRLEN];
  //char title[STRLEN];
  //strcpy(device, "normal.ps/ps");
  //strcpy(title,  "Normal Distribution");
  //plotit(histogram.data, min, max, device, title);
}
