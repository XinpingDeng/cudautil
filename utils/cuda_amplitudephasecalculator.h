#ifndef _CUDA_AMPLITUDEPHASECALCULATOR_H
#define _CUDA_AMPLITUDEPHASECALCULATOR_H

#pragma once

#include "cuda_shared_utils.h"
//! A template kernel to calculate phase and amplitude of input array 
/*!
 * 
 * \see scalar_typecast
 *
 * \tparam T Complex number component data type
 * 
 * \param[in]  v         input Complex data
 * \param[in]  nsamp     Number of data samples to be calculated
 * \param[out] amplitude Calculated amplitude
 * \param[out] phase     Calculated amplitude
 *
 */
template <typename T>
__global__ void amplitude_phase_calculator(const T *v, float *amplitude, float *phase, int nsamp){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(idx < nsamp){
    // We always do calculation in float
    float v1;
    float v2;
    
    scalar_typecast(v[idx].x, v1);
    scalar_typecast(v[idx].y, v2);
    
    amplitude[idx] = sqrtf(v1*v1+v2*v2);
    phase[idx]     = atan2f(v2, v1); // in radians
  }
}

template <typename T>
class AmplitudePhaseCalculator{
public:
  float *amp = NULL;///< Calculated amplitude on device
  float *pha = NULL;///< Calculated phase on device
  float *h_amp = NULL; ///< Calculated amplitude on host
  float *h_pha = NULL; ///< Calculated phase on host
  
  //! Constructor of AmplitudePhaseCalculator class.
  /*!
   * 
   * - initialise the class
   * - create device memory for amplitude and phase
   * - calculate phase and amplitude with CUDA
   *
   * \see amplitude_phase_calculator
   *
   * \tparam TIN Input data type
   * 
   * \param[in] raw  input Complex data
   * \param[in] nsamp   Number of samples to be calculated
   * \param[in] nthread Number of threads per CUDA block to run `amplitude_phase_calculator` kernel
   *
   */
  AmplitudePhaseCalculator(T *raw,
			   int nsamp,
			   int nthread,
			   int host
			   )
    :nsamp(nsamp), nthread(nthread), host(host){

    nbyte = nsamp*sizeof(float);
    
    // sourt out input data
    data = copy2device(raw, nsamp, type);
    
    // Get output buffer as managed
    checkCudaErrors(cudaMalloc(&amp, nbyte));
    checkCudaErrors(cudaMalloc(&pha, nbyte));
  
    // Get amplitude and phase
    nblock = ceil(nsamp/(float)nthread+0.5);
    amplitude_phase_calculator<<<nblock, nthread>>>(data, amp, pha, nsamp);

    remove_device_copy(type, data);

    if(host){
      checkCudaErrors(cudaMallocHost(&h_amp, nbyte));
      checkCudaErrors(cudaMallocHost(&h_pha, nbyte));

      checkCudaErrors(cudaMemcpy(h_amp, amp, nbyte));
      checkCudaErrors(cudaMemcpy(h_pha, pha, nbyte));
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealGeneratorNormal class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~AmplitudePhaseCalculator(){
    
    checkCudaErrors(cudaFree(amp));
    checkCudaErrors(cudaFree(pha));

    if(host){
      checkCudaErrors(cudFreeHost(amp));
      checkCudaErrors(cudFreeHost(pha));
    }
    checkCudaErrors(cudaDeviceSynchronize());
  }

private:
  int nsamp; ///< number of values as a private parameter
  int nblock; ///< Number of CUDA blocks
  int nthread; ///< number of threas per blocks
  int host; ///< marker to tell if need a copy on host 
  int nbyte; ///< Number of bytes of output
  
  T *data; ///< To get hold on the input data
};


#endif
