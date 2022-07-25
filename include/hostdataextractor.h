#ifndef _HOSTDATAEXTRACTOR_H
#define _HOSTDATAEXTRACTOR_H

#include "cudautil.h"
#include "helper_cuda.h"


/*! \brief A class to copy data from host to device
 *
 * \tparam T data type, which can be a complex data type
 * 
 */
template <typename T>
class HostDataExtractor {
public:
  T *d_data = NULL; ///< Device buffer to hold data

  /*! Constructor of class HostDataExtractor
   *
   * \param[in] h_data Host data 
   * \param[in] ndata Number of data on host as type \p T
   *
   */
  HostDataExtractor(T *h_data, int ndata)
    :h_data(h_data), ndata(ndata){
    
    size = ndata*sizeof(T);
    checkCudaErrors(cudaMalloc(&d_data, size));
    checkCudaErrors(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
  }

  /*! Overload of class HostDataExtractor
   *
   * \param[in] h_data Host data 
   * \param[in] ndata  Number of data on host as type \p T
   * \param[in] stream On which stream to copy data, only for async
   *
   */
  HostDataExtractor(T *h_data, int ndata, cudaStream_t stream)
    :h_data(h_data), ndata(ndata){
    
    size = ndata*sizeof(T);
    checkCudaErrors(cudaMalloc(&d_data, size));
    checkCudaErrors(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));
  }

  //! Deconstructor of HostDataExtractor class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~HostDataExtractor(){
    checkCudaErrors(cudaFree(d_data));
  }
  
private:
  T *h_data = NULL;
  int ndata;
  int size;
};

#endif
