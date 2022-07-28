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

int find_maxmin(float *data, float &datamin, float &datamax, int ndata){

  datamin = data[0];
  datamax = data[0];
  
  for(int i = 1; i < ndata; i++){
    datamin = (datamin>data[i]) ? data[i] : datamin;
    datamax = (datamax<data[i]) ? data[i] : datamax;
  }

  return EXIT_SUCCESS;
}

int create_x(float xmin, float xmax, int ndata, float *x){
  
  for(int i = 0; i < ndata; i++){
    x[i] = xmin + i*(xmax-xmin)/(float)ndata;
  }
  
  return EXIT_SUCCESS;
}

int create_y(unsigned *y_int, float *y_float, int ndata){
  
  for(int i = 0; i < ndata; i++){
    y_float[i] = y_int[i];
  }
  
  return EXIT_SUCCESS;
}

int plotit(unsigned *data, float min, float max, char *device, char *title){

  float x[NUM_BINS];
  float y[NUM_BINS];
  create_x(min, max, NUM_BINS, x);
  create_y(data, y, NUM_BINS);

  float ymax;
  float ymin;
  find_maxmin(y, ymin, ymax, NUM_BINS);
  
  /* Open graphics device. */
  if (cpgopen(device) < 1){
    //if (cpgopen("/xw") < 1){
    fprintf(stderr, "Can not open device to plot\n");
    exit(1);
  }

  /* Get rid of  Press RETURN for next page:  */
  cpgask(0);

  /* Axis ranges */
  cpgenv(x[0], x[NUM_BINS-1], ymin, ymax, 0, 0);

  /* Label the axes (note use of \\u and \\d for raising exponent). */
  cpglab("Sample Value", "Number of Samples", title);

  /* plot histogram */
  cpgpt(NUM_BINS, x, y, 1);

  /* Close plot figure */
  cpgclos();
  
  return EXIT_SUCCESS;
}

TEST_CASE("RealDataGeneratorUniform", "RealDataGeneratorUniform") {

  int nthread = 128;
  int ndata = 102400000;
  int exclude = 0;
  int include = 100;

  cudaEvent_t g_start;
  cudaEvent_t g_stop;
  float gtime = 0;
  checkCudaErrors(cudaEventCreate(&g_start));
  checkCudaErrors(cudaEventCreate(&g_stop));
  CUDA_STARTTIME(g);
  
  // Get data
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
  RealDataGeneratorUniform uniform_data(gen, ndata, exclude, include, nthread);
  print_cuda_memory_info();

  // Get mean and standard deviation
  RealDataMeanStddevCalcultor<float> mean_stddev(uniform_data.data, ndata, nthread, 7);
  cout << "uniform data mean is " << mean_stddev.mean << "\t"
       << "uniform data stddev is " << mean_stddev.stddev 
       << endl;

  // Get histogram
  float min = exclude;
  float max = include;
  int nblock = 256;
  RealDataHistogram<float> histogram(uniform_data.data, ndata, min, max, nblock, nthread);

  CUDA_STOPTIME(g);
  cout << "elapsed time is " << gtime << " milliseconds" << endl;
  
  // plot histogram
  char device[STRLEN];
  char title[STRLEN];
  strcpy(device, "uniform.ps/ps");
  strcpy(title,  "Uniform Distribution");
  plotit(histogram.data, min, max, device, title);
}    

TEST_CASE("RealDataGeneratorNormal", "RealDataGeneratorNormal") {

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
  
  // Get data
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
  RealDataGeneratorNormal normal_data(gen, mean, stddev, ndata);
  print_cuda_memory_info();

  // Get mean and standard deviation
  RealDataMeanStddevCalcultor<float> mean_stddev(normal_data.data, ndata, nthread, 7);
  cout << "normal data mean is " << mean_stddev.mean << "\t"
       << "normal data stddev is " << mean_stddev.stddev 
       << endl;

  // Get histogram
  float min = -50;
  float max = 50;
  int nblock = 256;
  RealDataHistogram<float> histogram(normal_data.data, ndata, min, max, nblock, nthread);

  CUDA_STOPTIME(g);
  cout << "elapsed time is " << gtime << " milliseconds" << endl;
  
  // plot histogram
  char device[STRLEN];
  char title[STRLEN];
  strcpy(device, "normal.ps/ps");
  strcpy(title,  "Normal Distribution");
  plotit(histogram.data, min, max, device, title);
}
