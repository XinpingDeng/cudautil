#ifndef _HOSTMEMORYALLOCATOR_H
#define _HOSTMEMORYALLOCATOR_H

#include "cudautil.h"
#include "helper_cuda.h"


/*! \brief A class to allocate memory on host 
 *
 * \tparam T data type of the host memory, which can be a complex data type
 * 
 */
template <typename T>
class HostMemoryAllocator {
 public:
  T *h_data = NULL; ///< Host memory
  
  /*! Constructor of class HostMemoryAllocator
   *
   * \param[in] ndata Number of data on host as type \p T
   *
   */
 HostMemoryAllocator(int ndata)
   :ndata(ndata){
    checkCudaErrors(cudaMallocHost(&h_data, ndata*sizeof(T)));
  }
  
  //! Deconstructor of HostMemoryAllocator class.
  /*!
   * 
   * - free host memory at the class life end
   */
  ~HostMemoryAllocator(){
    checkCudaErrors(cudaFreeHost(h_data));
  }

 private:
  int ndata; /// < Number of data points   
};


#endif
