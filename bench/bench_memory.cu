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
  
  // run it 
  for(int i = 0; i < nrepeat; i++){
    cudaEvent_t g_start;
    cudaEvent_t g_stop;
    float gtime = 0;
    checkCudaErrors(cudaEventCreate(&g_start));
    checkCudaErrors(cudaEventCreate(&g_stop));
  
    CUDA_STARTTIME(g);

    krnl_add<<<nblock, nthread>>>(d_in1, d_in2, d_out, ndata);
    getLastCudaError("Kernel execution failed [ krnl_add ]");

    CUDA_STOPTIME(g);
    std::cout << "elapsed time with device memory is " << gtime << " milliseconds" << std::endl;
  }
  
  // free memory
  checkCudaErrors(cudaFree(d_in1));
  checkCudaErrors(cudaFree(d_in2));
  checkCudaErrors(cudaFree(d_out));
  
  // get device memory space
  int device = -1;
  checkCudaErrors(cudaGetDevice(&device));
  std::cout << "device is " << device << std::endl;
  
  float *m_in1 = NULL;
  float *m_in2 = NULL;
  float *m_out = NULL;
  checkCudaErrors(cudaMallocManaged(&m_in1, nbyte, cudaMemAttachGlobal));
  checkCudaErrors(cudaMallocManaged(&m_in2, nbyte, cudaMemAttachGlobal));
  checkCudaErrors(cudaMallocManaged(&m_out, nbyte, cudaMemAttachGlobal));

  // setup device memory
  checkCudaErrors(cudaMemcpy(m_in1, data.data, nbyte, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(m_in2, data.data, nbyte, cudaMemcpyHostToDevice));
  
  // fetch to get bench right, this is important to make bench with managed memory right
  checkCudaErrors(cudaMemPrefetchAsync(m_in1, nbyte, device, NULL));
  checkCudaErrors(cudaMemPrefetchAsync(m_in2, nbyte, device, NULL));
  checkCudaErrors(cudaMemPrefetchAsync(m_out, nbyte, device, NULL));
 
  // run it 
  for(int i = 0; i < nrepeat; i++){
    cudaEvent_t g_start;
    cudaEvent_t g_stop;
    float gtime = 0;
    checkCudaErrors(cudaEventCreate(&g_start));
    checkCudaErrors(cudaEventCreate(&g_stop));
  
    CUDA_STARTTIME(g);

    krnl_add<<<nblock, nthread>>>(m_in1, m_in2, m_out, ndata);
    getLastCudaError("Kernel execution failed [ krnl_add ]");

    checkCudaErrors(cudaDeviceSynchronize());
    
    CUDA_STOPTIME(g);
    std::cout << "elapsed time with managed memory is " << gtime << " milliseconds" << std::endl;
  }
  
  // free memory
  checkCudaErrors(cudaFree(m_in1));
  checkCudaErrors(cudaFree(m_in2));
  checkCudaErrors(cudaFree(m_out));
  
  return EXIT_SUCCESS;
}
