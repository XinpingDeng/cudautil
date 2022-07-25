#ifndef _COMPLEXDATAGENERATOR_H
#define _COMPLEXDATAGENERATOR_H

#include "cudautil.h"
#include "helper_cuda.h"

//! A template kernel to build complex numbers with its real and imag part
/*!
 * 
 * \see scalar_typecast
 *
 * \tparam TREAL Real part data type
 * \tparam TIMAG Imag part data type
 * \tparam TCMPX So-called complex data type, which is actually the data type of its components
 * 
 * \param[in]  d_real  Real part to build complex numbers
 * \param[in]  d_imag  Imag part to build complex numbers
 * \param[in]  ndata   Number of data points ton be built
 * \param[out] d_cmpx  Complex numbers
 *
 */
template <typename TREAL, typename TIMAG, typename TCMPX>
__global__ void g_complexbuilder(const TREAL *d_real, const TIMAG *d_imag, TCMPX *d_cmpx, int ndata){
  // Maximum x-dimension of a grid of thread blocks is 2^31-1
  // Maximum x- or y-dimension of a block is 1024
  // So here we can cover (2^31-1)*1024 random numbers, which are 2^41-1024
  // should be big enough
  
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<ndata){

    scalar_typecast(d_real[idx], d_cmpx[2*idx]);
    scalar_typecast(d_imag[idx], d_cmpx[2*idx+1]);
  }
}

/*! \brief A class to build a complex vector with two real vectors
 *
 * \tparam TREAL Typename of the real part data
 * \tparam TIMAG Typename of the imag part data
 * \tparam TCMPX Typename of the complex data
 *
 * The class does not really build a complex vector, instead it converts \p d_real and \p d_imag to \p TCMPX
 * and then put the data interleaved as [REAL IMAG ... REAL IMAG], so that we can cast the number to complex data type later.
 * As of that the data type \tparam TCMPX is not a complex data type, but the data type of its components.
 *
 * The class use kernel `g_complexbuilder` to convert data type and build complex numbers. `g_complexbuilder` uses `scalar_typecast` to convert data type. 
 * As of that, the allowed data type here is limited to following table (more types can be added later) 
 * 
 * TREAL/TIMAG    | TCMPX
 * -------|----
 * float  | float
 * float  | double
 * float  | half 
 * float  | int
 * float  | int16_t
 * float  | int8_t
 * double | float
 * half   | float
 * int    | float
 * int16_t| float
 * int8_t | float
 *
 */
template <typename TREAL, typename TIMAG, typename TCMPX>
class ComplexDataBuilder {
public:
  TCMPX *d_cmpx = NULL; ///< A so-called complex data on device with data type \tparam TCMPX
  
  //! Constructor of ComplexDataBuilder class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata complex numbers
   * - build complex numbers with \p d_real and \p d_imag
   *
   * \see g_complexbuilder, scalar_typecast
   *
   * \tparam TREAL Real part data type
   * \tparam TIMAG Imag part data type
   * \tparam TCMPX So-called complex data type, which is actually the data type of its components
   * 
   * \param[in] d_real  Real part to build complex numbers
   * \param[in] d_imag  Imag part to build complex numbers
   * \param[in] ndata   Number of data points ton be converted
   * \param[in] nthread Number of threads per CUDA block to run `g_complexbuilder` kernel
   *
   */
  ComplexDataBuilder(TREAL *d_real, TIMAG *d_imag, int ndata, int nthread )
    :d_real(d_real), d_imag(d_imag), ndata(ndata), nthread(nthread){
    checkCudaErrors(cudaMalloc(&d_cmpx, 2*ndata*sizeof(TCMPX)));

    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;
    
    g_complexbuilder<<<nblock, nthread>>>(d_real, d_imag, d_cmpx, ndata);
    getLastCudaError("Kernel execution failed [ g_complexbuilder ]");
  }
  
  //! Deconstructor of ComplexDataBuilder class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~ComplexDataBuilder(){
    checkCudaErrors(cudaFree(d_cmpx));
  }
  
private:
  TREAL *d_real = NULL;
  TIMAG *d_imag = NULL;
  
  int ndata;
  int nthread;
  int nblock;
};
#endif
