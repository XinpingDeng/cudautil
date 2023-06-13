#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "utils/cuda_utils.h"

using namespace std;

// We can not compare converted numbers against CPU implementation
// We can only check the difference between original data and converted data with mean and standard deviation
int main(int argc, char *argv[]){

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
  RealGeneratorNormal normal_data(gen, mean, stddev, ndata);
  print_cuda_memory_info();

  // Convert to float
  RealConvertor<float, float> normal_data_float(normal_data.data, ndata, nthread);

  // Get the difference
  RealDifferentiator<float, float> normal_data_diff(normal_data.data, normal_data_float.data, ndata, nthread);
  
  // Get mean and standard deviation
  RealMeanStddevCalculator<float> mean_stddev(normal_data_diff.data, ndata, nthread, 7);
  
  CUDA_STOPTIME(g);
  cout << "elapsed time is " << gtime << " milliseconds" << endl;

  float mean_f = mean_stddev.mean;
  float stddev_f = mean_stddev.stddev ;
  cout << "mean is " << mean_f << " "
       << "stddev is " << stddev_f 
       << endl;

  return EXIT_SUCCESS;
}
