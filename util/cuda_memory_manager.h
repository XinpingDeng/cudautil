// cuda_memory_manager.h
//
// last-edit-by: <> 
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#ifndef CUDA_MEMORY_MANAGER_H
#define CUDA_MEMORY_MANAGER_H 1

#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>

inline int print_cuda_memory_info() {
  //cudaError_t status;
  size_t free, total;
  
  checkCudaErrors(cudaMemGetInfo(&free, &total));
  
  fprintf(stdout, "GPU free memory is %.1f, total is %.1f MBbytes\n",
	  free/1024.0/1024, total/1024.0/1024);
  
  if(free<=0){
    fprintf(stderr, "Use too much GPU memory.\n");
    exit(EXIT_FAILURE);
  }
  
  return EXIT_SUCCESS;
}

/*! A template function to get a buffer into device
  It check where is the input buffer, if the buffer is on device, it just pass the pointer
  otherwise it alloc new buffer on device and copy the data to it
*/
template <typename T>
T* copy2device(T *raw, int ndata, enum cudaMemoryType &type){
  T *data = NULL;
  
  cudaPointerAttributes attributes; ///< to hold memory attributes
  
  // cudaMemoryTypeUnregistered for unregistered host memory,
  // cudaMemoryTypeHost for registered host memory,
  // cudaMemoryTypeDevice for device memory or
  // cudaMemoryTypeManaged for managed memory.
  checkCudaErrors(cudaPointerGetAttributes(&attributes, raw));
  type = attributes.type;
  
  if(type == cudaMemoryTypeUnregistered || type == cudaMemoryTypeHost){
    int nbytes = ndata*sizeof(T);
    checkCudaErrors(cudaMallocManaged(&data, nbytes, cudaMemAttachGlobal));
    checkCudaErrors(cudaMemcpy(data, raw, nbytes, cudaMemcpyDefault));
  }
  else{
    data = raw;
  }
  
  return data;
}

/*! A function to free device memory if it is a copy of a host memory
 */
template<typename T>
int remove_device_copy(enum cudaMemoryType type, T *data){
  
  if(type == cudaMemoryTypeUnregistered || type == cudaMemoryTypeHost){
    checkCudaErrors(cudaFree(data));
  }
  
  return EXIT_SUCCESS;
}

/*! \brief A class to allocate memory on host 
 *
 * \tparam T data type of the host memory, which can be a complex data type
 * 
 */
template <typename T>
class HostMemoryAllocator {
public:
  T *data = NULL; ///< Host memory or managed memory
  
  /*! Constructor of class HostMemoryAllocator
   *
   * \param[in] ndata  Number of data on host as type \p T
   * \param[in] device Marker to tell if we also need a copy on device
   *
   */
  HostMemoryAllocator(int ndata, int device=0)
    :ndata(ndata), device(device){
    if(device){
      checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(T), cudaMemAttachGlobal));
    }
    else{
      checkCudaErrors(cudaMallocHost(&data, ndata*sizeof(T)));
    }
  }
  
  //! Deconstructor of HostMemoryAllocator class.
  /*!
   * 
   * - free host memory at the class life end
   */
  ~HostMemoryAllocator(){
    checkCudaErrors(cudaFreeHost(data));
    if(device){
      checkCudaErrors(cudaFree(data));
    }
  }

private:
  int ndata; ///< Number of data points
  int device; ///< Do we need a copy on device?
};

/*! \brief A class to allocate memory on device 
 *
 * \tparam T data type of the device memory, which can be a complex data type
 * 
 */
template <typename T>
class DeviceMemoryAllocator {
public:
  T *data = NULL; ///< Device memory or managed memory
  
  /*! Constructor of class DeviceMemoryAllocator
   *
   * \param[in] ndata Number of data on host as type \p T
   * \param[in] host  Marker to see if we also need a copy on host
   *
   */
  DeviceMemoryAllocator(int ndata, int host=0)
    :ndata(ndata), host(host){
    if(host){
      checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(T), cudaMemAttachGlobal));
    }
    else{
      checkCudaErrors(cudaMalloc(&data, ndata*sizeof(T)));
    }
  }
  
  //! Deconstructor of DeviceMemoryAllocator class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~DeviceMemoryAllocator(){
    checkCudaErrors(cudaFree(data));
  }

private:
  int ndata; ///< Number of data points
  int host;  ///< Do we also need to copy on host?
};

/*! \brief A class to copy data from host to device
 *
 * \tparam T data type, which can be a complex data type
 * 
 */
template <typename T>
class HostDataExtractor {
public:
  T *data = NULL; ///< Device buffer to hold data

  /*! Constructor of class HostDataExtractor
   *
   * \param[in] h_data Host data 
   * \param[in] ndata Number of data on host as type \p T
   * async memcpy will not work here as we always get new copy of memory
   */
  HostDataExtractor(T *h_data, int ndata)
    :h_data(h_data), ndata(ndata){
    
    size = ndata*sizeof(T);
    checkCudaErrors(cudaMalloc(&data, size));
    checkCudaErrors(cudaMemcpy(data, h_data, size, cudaMemcpyDefault));
  }
  
  //! Deconstructor of HostDataExtractor class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~HostDataExtractor(){
    checkCudaErrors(cudaFree(data));
  }
  
private:
  T *h_data = NULL;
  int ndata;
  int size;
};

/*! \brief A class to copy device data to host
 *
 * \tparam T data type, which can be a complex data type
 * 
 */
template <typename T>
class DeviceDataExtractor {
public:
  T *data = NULL; ///< Host buffer to hold data
  
  /*!
   * \param[in] d_data Device data 
   * \param[in] ndata Number of data on host as type \p T
   * async memcpy will not work here as we always get new copy of memory
   */
  DeviceDataExtractor(T *d_data, int ndata)
    :d_data(d_data), ndata(ndata){
   
    size = ndata*sizeof(T);
    checkCudaErrors(cudaMallocHost(&data, size));
    checkCudaErrors(cudaMemcpy(data, d_data, size, cudaMemcpyDefault));
  }
  
  //! Deconstructor of DeviceDataExtractor class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~DeviceDataExtractor(){
    checkCudaErrors(cudaFreeHost(data));
  }
  
private:
  T *d_data = NULL;
  int ndata;
  int size;
};


/*! \brief A class to allocate memory as managed
 *
 * \tparam T data type of the host memory, which can be a complex data type
 * 
 */
template <typename T>
class ManagedMemoryAllocator {
public:
  T *data = NULL; ///< Managed memory 
  
  /*! Constructor of class ManagedMemoryAllocator
   *
   * \param[in] ndata  Number of data on host as type \p T
   *
   */
  ManagedMemoryAllocator(int ndata)
    :ndata(ndata){
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(T), cudaMemAttachGlobal));
  }
  
  //! Deconstructor of ManagedMemoryAllocator class.
  /*!
   * 
   * - free host memory at the class life end
   */
  ~ManagedMemoryAllocator(){
    checkCudaErrors(cudaFree(data));
  }
  
private:
  int ndata; ///< Number of data points
};

#endif // CUDA_MEMORY_MANAGER_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//

