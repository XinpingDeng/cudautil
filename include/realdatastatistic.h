#ifndef _REALDATASTATISTIC_H
#define _REALDATASTATISTIC_H

#include "cudautil.h"
#include "realdatareduction.h"
#include "helper_cuda.h"

/*! \brief A function to convert input data from \p T to float and calculate its power in parallel on GPU
 * 
 * It is a template function to convert input data from \p T to float and calculate its power in parallel on GPU
 * \tparam T  The input data type
 *
 * The data type convertation is done with an overloadded function `scalar_typecast`
 * \see scalar_typecast
 *
 * The supported data convertation is shown in the following table (we can add more support here later)
 *
 * |T      |
 * |-------|
 * |double |
 * |half   |
 * |int    |
 * |int16_t|
 * |int8_t |
 *
 * \param[in]  d_data   Input data
 * \param[in]  ndata    Number of data
 * \param[out] d_float  Converted data in float
 * \param[out] d_float2 Power of converted data 
 *
 */
template <typename T>
__global__ void g_pow(const T *d_data, float *d_float, float *d_float2, int ndata){
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  if(idx < ndata){
    float f_data;

    scalar_typecast(d_data[idx], f_data);
    d_float[idx]  = f_data;
    d_float2[idx] = f_data*f_data;
  }
}

template <typename T>
class RealDataMeanStddevCalcultor {
  
public:
  float mean;   ///< Mean of the difference between input two vectors, always in float
  float stddev; ///< Standard deviation of the difference between input two vectors, always in float

  
  //! Constructor of class RealDataMeanStddevCalcultor
  /*!
   * - initialise the class
   * - create required device memory
   * - convert input data from \p T to float and calculate its power in a single CUDA kernel `g_pow`
   * - reduce the float data and its power to get mean
   * - calculate standard deviation with the mean of float and power data
   *
   * \param[in] d_data  The input vector on device with data type \p T
   * \param[in] ndata   Number of data
   * \param[in] nthread Number of threads per CUDA block to run kernel `g_pow`
   * \param[in] method  Data reduction method, which can be from 0 to 7 inclusive
   *
   * As kernel `g_pow` uses `scalar_typecast` to convert \p T to float, the support \p T can be
   *
   * |T |
   * |--|
   * |double | 
   * |half   |
   * |int    | 
   * |int16_t|
   * |int8_t |
   * 
   * \see g_pow, reduce, scalar_typecast
   *
   */
  RealDataMeanStddevCalcultor(T *d_data, int ndata, int nthread, int method)
    :d_data(d_data), ndata(ndata), nthread(nthread), method(method){
    
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;
    
    checkCudaErrors(cudaMalloc(&d_float, ndata*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_float2, ndata*sizeof(float)));

    checkCudaErrors(cudaMalloc(&d_reduction, nblock*sizeof(float)));
    checkCudaErrors(cudaMallocHost(&d_sum, sizeof(float)));
    checkCudaErrors(cudaMallocHost(&d_sum2, sizeof(float)));

    g_pow<<<nblock, nthread>>>(d_data, d_float, d_float2, ndata);
    getLastCudaError("Kernel execution failed [ g_pow ]");
    
    // First reduce mean data
    reduce(ndata,  nthread, nblock, method, d_float, d_reduction);
    if(nblock > 1){
      reduce(nblock, nthread, 1, method, d_reduction, d_float);
      checkCudaErrors(cudaMemcpy(d_sum, d_float, sizeof(float), cudaMemcpyDeviceToHost));
    }else{
      checkCudaErrors(cudaMemcpy(d_sum, d_reduction, sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    // Second reduce mean power 2 data
    reduce(ndata,  nthread, nblock, method, d_float2, d_reduction);
    if(nblock > 1){
      reduce(nblock, nthread, 1, method, d_reduction, d_float2);
      checkCudaErrors(cudaMemcpy(d_sum2, d_float2, sizeof(float), cudaMemcpyDeviceToHost));
    }else{
      checkCudaErrors(cudaMemcpy(d_sum2, d_reduction, sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    mean  = d_sum[0]/(float)ndata;
    mean2 = d_sum2[0]/(float)ndata;
    
    stddev = sqrtf(mean2 - mean*mean);
  }
  
  //! Deconstructor of RealDataMeanStddevCalcultor class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealDataMeanStddevCalcultor(){
    checkCudaErrors(cudaFree(d_float));
    checkCudaErrors(cudaFree(d_float2));
    checkCudaErrors(cudaFree(d_reduction));
    
    checkCudaErrors(cudaFreeHost(d_sum));
    checkCudaErrors(cudaFreeHost(d_sum2));
  }
  
private:

  int ndata; ///< Number of input data
  int nthread; ///< Number of threads per CUDA block
  int nblock;  ///< Number of CUDA blocks
  int method; ///< data d_reduction method
  
  T *d_data = NULL;
  float *d_float = NULL;
  float *d_float2 = NULL;

  float *d_reduction; ///< it holds intermediate float data duration data d_reduction on device
  
  float *d_sum;  ///< d_sum of difference
  float *d_sum2; ///< d_sum of difference power of 2
  float mean2; ///< mean of difference power of 2
};

#endif
