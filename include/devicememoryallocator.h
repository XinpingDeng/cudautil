#ifndef _DEVICEMEMORYALLOCATOR_H
#define _DEVICEMEMORYALLOCATOR_H

#include "cudautil.h"
#include "helper_cuda.h"

/*! \brief A class to allocate memory on device 
 *
 * \tparam T data type of the device memory, which can be a complex data type
 * 
 */
template <typename T>
class DeviceMemoryAllocator {
 public:
  T *d_data = NULL; ///< Device memory
  
  /*! Constructor of class DeviceMemoryAllocator
   *
   * \param[in] ndata Number of data on host as type \p T
   *
   */
 DeviceMemoryAllocator(int ndata)
   :ndata(ndata){
    checkCudaErrors(cudaMalloc(&d_data, ndata*sizeof(T)));
  }
  
  //! Deconstructor of DeviceMemoryAllocator class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~DeviceMemoryAllocator(){
    checkCudaErrors(cudaFree(d_data));
  }

 private:
  int ndata; /// < Number of data points   
};

#endif
