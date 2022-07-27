#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/cudautil.h"
#include "../include/reduction_kernel.h"

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

/*! A kernel to contraint random number from range (0.0 1.0] to range (exclude include] or [include exclude).
 *
 * \param[in, out] data    The input data in range (0.0 1.0] and new data in range (exclude include] or [include exclude) is also returned with it.
 * \param[in]      exclude The exclusive end of random numbers
 * \param[in]      range   The range of random numbers, it does not have to be positive, it is calculated with `include - exclude`
 * \param[in]      ndata   Number of data
 *
 */
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
