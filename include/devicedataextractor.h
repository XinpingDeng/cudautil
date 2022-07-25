#ifndef _DEVICEDATAEXTRACTOR_H
#define _DEVICEDATAEXTRACTOR_H

#include "cudautil.h"
#include "helper_cuda.h"

/*! \brief A class to copy device data to host
 *
 * \tparam T data type, which can be a complex data type
 * 
 */
template <typename T>
class DeviceDataExtractor {
 public:
  T *h_data = NULL; ///< Host buffer to hold data
  
  /*!
   * \param[in] d_data Device data 
   * \param[in] ndata Number of data on host as type \p T
   *
   */
 DeviceDataExtractor(T *d_data, int ndata)
   :d_data(d_data), ndata(ndata){
   
    size = ndata*sizeof(T);
    checkCudaErrors(cudaMallocHost(&h_data, size));
    checkCudaErrors(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
  }
  
  /*!
   * \param[in] d_data Device data 
   * \param[in] ndata  Number of data on host as type \p T
   * \param[in] stream On which stream to copy data, only for async
   *
   */
  DeviceDataExtractor(T *d_data, int ndata, cudaStream_t stream)
    :d_data(d_data), ndata(ndata){
    
    size = ndata*sizeof(T);
    checkCudaErrors(cudaMallocHost(&h_data, size));
    checkCudaErrors(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream));
  }

  //! Deconstructor of DeviceDataExtractor class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~DeviceDataExtractor(){
    checkCudaErrors(cudaFreeHost(h_data));
  }
  
 private:
  T *d_data = NULL;
  int ndata;
  int size;
};

#endif
