#ifndef _REALDATADIFFERENTIATOR_H
#define _REALDATADIFFERENTIATOR_H

#include "cudautil.h"
#include "helper_cuda.h"

/*! \brief Overloadded kernel to get d_difference between two real input vectors
 *
 * \tparam T1 Data type of the first input vector
 * \tparam T2 Data type of the second input vector
 * 
 * \param[in]  d_data1 The first input vector in \p T1
 * \param[in]  d_data2 The second input vector in \p T2
 * \param[in]  ndata   Number of data
 * \param[out] d_diff  The d_difference between these two vectors in float, it is always in float
 *
 * The kernel uses `scalar_subtract` to get difference (in float) between two numbers and currently it supports (we can add more support later).
 *
 * T1     | T2
 * -------|----
 * float  | float
 * float  | half
 * half   | float 
 * half   | half 
 * 
 * \see scalar_subtract
 * 
 */
template <typename T1, typename T2>
__global__ void g_subtract(const T1 *d_data1, const T2 *d_data2, float *d_diff, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;

  if(idx < ndata){
    scalar_subtract(d_data1[idx], d_data2[idx], d_diff[idx]);
  }
}

/*! \brief A class to get the difference between two real vectors
 *
 * \tparam T1 Typename of the data in one vector
 * \tparam T2 Typename of the data in the other vector
 *
 * 
 * Suggested combinations of T1 and T2 are (other combinations may not work, we can add more support later)
 * T1     | T2
 * -------|----
 * float  | float
 * float  | half
 * half   | float 
 * half   | half  
 * 
 * The class to get difference between two real vectors, it is allowed to have different types for these inputs and
 * the result will be in float.
 * 
 */
template <typename T1, typename T2>
class RealDataDifferentiator {

public:
  float *d_diff  = NULL;  ///< the difference between input \p d_data1 and \p d_data2
  
  //! Constructor of RealDataDifferentiator class.
  /*!
   * 
   * - initialise the class
   * - create device memory for the difference \p d_diff
   * - calculate the difference with a CUDA kernel `g_subtract`
   *
   * \see g_subtract, scalar_subtract
   * 
   * \param[in] d_data1 The first input real vector
   * \param[in] d_data2 The second input real vector
   * \param[in] ndata   Number of float random numbers to generate
   * \param[in] nthread Number of threads per CUDA block to run `g_convert`
   *
   */
  RealDataDifferentiator(T1 *d_data1, T2 *d_data2, int ndata, int nthread)
    :d_data1(d_data1), d_data2(d_data2), ndata(ndata), nthread(nthread){
    
    // Now get memory for \p diff 
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;
    checkCudaErrors(cudaMalloc(&d_diff,  ndata*sizeof(float)));
    
    // Now get difference
    g_subtract<<<nblock, nthread>>>(d_data1, d_data2, d_diff, ndata);
    getLastCudaError("Kernel execution failed [ g_subtract ]");
  }
  
  //! Deconstructor of RealDataDifferentiator class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealDataDifferentiator(){
    
    checkCudaErrors(cudaFree(d_diff));
  }
  
private:

  int ndata; ///< Number of input data
  int nthread; ///< Number of threads per CUDA block
  int nblock;  ///< Number of CUDA blocks

  T1 *d_data1 = NULL; ///< private variable to hold input vector pointer 1
  T2 *d_data2 = NULL; ///< private variable to hold input vector pointer 2
};

#endif
