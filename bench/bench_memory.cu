#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "utils/cuda_utils.h"

__global__ void krnl_add(const float *d_in1, float *d_in2, float *d_out, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(idx < ndata){
    d_out[idx] = d_in1[idx] + d_in2[idx];
  }
}

int main(int argc, char *argv[]){

  int ndata    = 102400000;
  float mean   = 0;
  float stddev = 10;
  int nthread  = 128;
  int nrepeat  = 1000;
  
  std::cout << "ndata   " << ndata   << std::endl;
  std::cout << "mean    " << mean    << std::endl;
  std::cout << "stddev  " << stddev  << std::endl;
  std::cout << "nthread " << nthread << std::endl;
  std::cout << "nrepeat " << nrepeat << std::endl;

  uint64_t nbyte = ndata*sizeof(float);
  int nblock = ndata/nthread;
  
  std::cout << "buffer size is " << nbyte/1E9 << "Gbyte" << std::endl;
  std::cout << "nblock is      " << nblock << std::endl;
  
  // Get float data 
  curandGenerator_t gen;
  checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
  RealGeneratorNormal data(gen, mean, stddev, ndata);
  print_cuda_memory_info();

  // get device memory space
  float *d_in1 = NULL;
  float *d_in2 = NULL;
  float *d_out = NULL;
  checkCudaErrors(cudaMalloc(&d_in1, nbyte));
  checkCudaErrors(cudaMalloc(&d_in2, nbyte));
  checkCudaErrors(cudaMalloc(&d_out, nbyte));

  // setup device memory
  checkCudaErrors(cudaMemcpy(d_in1, data.data, nbyte, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_in2, data.data, nbyte, cudaMemcpyHostToDevice));
  
  // setup timer 
  cudaEvent_t g_start;
  cudaEvent_t g_stop;
  float gtime = 0;
  checkCudaErrors(cudaEventCreate(&g_start));
  checkCudaErrors(cudaEventCreate(&g_stop));
  
  CUDA_STARTTIME(g);
  for(int i = 0; i < nrepeat; i++){
    krnl_add<<<nblock, nthread>>>(d_in1, d_in2, d_out, ndata);
    getLastCudaError("Kernel execution failed [ krnl_add ]");

  }
  
  CUDA_STOPTIME(g);
  std::cout << "elapsed time is " << gtime/nrepeat << " milliseconds" << std::endl;

  // free memory
  checkCudaErrors(cudaFree(d_in1));
  checkCudaErrors(cudaFree(d_in2));
  checkCudaErrors(cudaFree(d_out));
  
  return EXIT_SUCCESS;
}
