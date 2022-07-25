#ifndef _AMPLITUDEPHASECALCULATOR_H
#define _AMPLITUDEPHASECALCULATOR_H

#include "cudautil.h"
#include "helper_cuda.h"

//! A template kernel to calculate phase and amplitude of input array 
/*!
 * 
 * \see scalar_typecast
 *
 * \tparam T Complex number component data type
 * 
 * \param[in]  v         Input real data which is the components of complex samples
 * \param[in]  ndata     Number of data samples to be calculated
 * \param[out] amplitude Calculated amplitude
 * \param[out] phase     Calculated amplitude
 *
 */
template <typename T>
__global__ void g_amplitude_phase(const T *v, float *amplitude, float *phase, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(idx < ndata){
    // We always do calculation in float
    float v1;
    float v2;
    
    scalar_typecast(v[2*idx],  v1);
    scalar_typecast(v[2*idx+1],v2);
    
    amplitude[idx] = sqrtf(v1*v1+v2*v2);
    phase[idx]     = atan2f(v2, v1); // in radians
  }
}

template <typename T>
class AmplitudePhaseCalculator{
public:
  float *d_amp = NULL;///< Calculated amplitude on device
  float *d_pha = NULL;///< Calculated phase on device
  
  //! Constructor of AmplitudePhaseCalculator class.
  /*!
   * 
   * - initialise the class
   * - create device memory for amplitude and phase
   * - calculate phase and amplitude with CUDA
   *
   * \see g_amplitude_phase
   *
   * \tparam TIN Input data type
   * 
   * \param[in] d_data  Data to be calculated, it is interleaved as [REAL IMAG ... REAL IMAG] as type \p T
   * \param[in] ndata   Number of samples to be converted, the size of d_data is 2*ndata
   * \param[in] nthread Number of threads per CUDA block to run `g_amplitude_phase` kernel
   *
   */
  AmplitudePhaseCalculator(T *d_data,
			   int ndata,
			   int nthread
			   )
    :d_data(d_data), ndata(ndata), nthread(nthread){
    // Get other buffers
    checkCudaErrors(cudaMalloc(&d_amp, ndata * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pha, ndata * sizeof(float)));
  
    // Get amplitude and phase
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;
    g_amplitude_phase<<<nblock, nthread>>>(d_data, d_amp, d_pha, ndata);
  }
  
  //! Deconstructor of RealDataGeneratorNormal class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~AmplitudePhaseCalculator(){
    
    checkCudaErrors(cudaFree(d_amp));
    checkCudaErrors(cudaFree(d_pha));
  }

private:
  int ndata; ///< number of values as a private parameter
  int nblock; ///< Number of CUDA blocks
  int nthread; ///< number of threas per block
  
  T *d_data; ///< To get hold on the input data
};

#endif
