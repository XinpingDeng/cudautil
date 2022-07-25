#ifndef _REALDATACONVERTOR_H
#define _REALDATACONVERTOR_H

#include "cudautil.h"
#include "helper_cuda.h"

/*! \brief A function to convert real data from \p TIN to \p TOUT on GPU
 * 
 * It is a template function which converts data from \p TIN to \p TOUT, where
 * \tparam TIN  The input data type
 * \tparam TOUT The output data type
 *
 * The data type convertation is done with an overloadded function `scalar_typecast`
 * \see scalar_typecast
 *
 * The supported data convertation is shown in the following table (we can add more support here later)
 *
 * TIN    | TOUT
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
 * \param[in]  input  Data to be converted
 * \param[in]  ndata  Number of data points to be converted
 * \param[out] output Converted data
 *
 */
template <typename TIN, typename TOUT>
__global__ void g_convert(const TIN *input, TOUT *output, int ndata){
  // Maximum x-dimension of a grid of thread blocks is 2^31-1
  // Maximum x- or y-dimension of a block is 1024
  // So here we can cover (2^31-1)*1024 random numbers, which are 2^41-1024
  // should be big enough
  
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<ndata){
    // Just in case we have a very small ndata
    scalar_typecast(input[idx], output[idx]);
  }
}

/*! \brief A class to convert real device data from one type \p TIN to another \p TOUT
 * 
 * It is a template class which converts data from \p TIN to \p TOUT, where
 * \tparam TIN  The input data type
 * \tparam TOUT The output data type
 *
 * The data type convertation is done with a template CUDA kernel function `g_convert`
 * \see g_convert
 *
 * The supported data convertation is shown in the following table (we can add more support here later)
 *
 * TIN    | TOUT
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
template <typename TIN, typename TOUT>
class RealDataConvertor{
public:
  TOUT *d_converted_data = NULL; ///< Converted data on device as \p TOUT data type
  
  //! Constructor of RealDataConvertor class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata float random numbers
   * - convert input data from \p TIN to \p TOUT on GPU with CUDA Kernel `g_convert`
   *
   * \see g_convert
   *
   * \tparam TIN Input data type
   * 
   * \param[in] d_data  Data to be converted with data type \p TIN on device
   * \param[in] ndata   Number of data points ton be converted
   * \param[in] nthread Number of threads per CUDA block to run `g_convert` kernel
   *
   */
  RealDataConvertor(TIN *d_data, int ndata, int nthread)
    :d_data(d_data), ndata(ndata), nthread(nthread){
    
    checkCudaErrors(cudaMalloc(&d_converted_data, ndata*sizeof(TOUT)));
    
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;

    g_convert<<<nblock, nthread>>>(d_data, d_converted_data, ndata);
    getLastCudaError("Kernel execution failed [ g_convert ]");
  }

  
  //! Deconstructor of RealDataGeneratorNormal class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealDataConvertor(){
  
    checkCudaErrors(cudaFree(d_converted_data));
  }
  
private:
  TIN *d_data = NULL; ///< An internal pointer to input data
  int ndata;   ///< Number of generated data
  int nthread; ///< Number of threads per CUDA block
  int nblock;  ///< Number of blocks to process \p ndata
};

#endif
