#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "cudautil.h"
#include "reduction_kernel.h"

int print_cuda_memory_info() {
  //cudaError_t status;
  size_t free, total;
  
  checkCudaErrors(cudaMemGetInfo(&free, &total));
  
  fprintf(stdout, "GPU free memory is %.1f, total is %.1f MBbytes\n",
	  free/1024.0/1024, total/1024.0/1024);
  
  if(free<=0){
    fprintf(stderr, "Use too much GPU memory.\n");
    exit(EXIT_FAILURE);
  }
  
  return EXIT_SUCCESS;
}

__global__ void cudautil_contraintor(float *data, float exclude, float range, int ndata){
  // Maximum x-dimension of a grid of thread blocks is 2^31-1
  // Maximum x- or y-dimension of a block is 1024
  // So here we can cover (2^31-1)*1024 random numbers, which are 2^41-1024
  // should be big enough

  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<ndata){
    // Just in case we have a very small ndata
    data[idx] = data[idx]*range+exclude;
  }
}
