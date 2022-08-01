// cuda_convertors.h
//
// last-edit-by: <> 
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#pragma once

#ifndef CUDA_CONVERTORS_H
#define CUDA_CONVERTORS_H 1

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
  __global__ void cudautil_convert(const TIN *input, TOUT *output, int ndata){
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
 * The data type convertation is done with a template CUDA kernel function `cudautil_convert`
 * \see cudautil_convert
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
  class RealConvertor{
 public:
  TOUT *data = NULL; ///< Converted data on Unified memory
  
  //! Constructor of RealConvertor class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata float random numbers
   * - convert input data from \p TIN to \p TOUT on GPU with CUDA Kernel `cudautil_convert`
   *
   * \see cudautil_convert
   *
   * \tparam TIN Input data type
   * 
   * \param[in] raw     Data to be converted with data type \p TIN on device or host
   * \param[in] ndata   Number of data points ton be converted
   * \param[in] nthread Number of threads per CUDA block to run `cudautil_convert` kernel
   *
   */
 RealConvertor(TIN *raw, int ndata, int nthread)
   :ndata(ndata), nthread(nthread){

    input = copy2device(raw, ndata, type);
    
    // Create output buffer as managed
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(TOUT), cudaMemAttachGlobal));

    // Setup kernel size and run it to convert data
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;    
    cudautil_convert<<<nblock, nthread>>>(input, data, ndata);
    getLastCudaError("Kernel execution failed [ cudautil_convert ]");

    // Free intermediate memory
    remove_device_copy(type, input);
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of RealConvertor class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~RealConvertor(){  
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
 private:
  enum cudaMemoryType type; ///< memory type
  TIN *input = NULL; ///< An internal pointer to input data
  int ndata;   ///< Number of generated data
  int nthread; ///< Number of threads per CUDA block
  int nblock;  ///< Number of blocks to process \p ndata
};


//! A template kernel to build complex numbers with its real and imag part
/*!
 * 
 * \see scalar_typecast
 *
 * \tparam TREAL Real part data type
 * \tparam TIMAG Imag part data type
 * \tparam TCMPX Complex data type
 * 
 * \param[in]  d_real  Real part to build complex numbers
 * \param[in]  d_imag  Imag part to build complex numbers
 * \param[in]  ndata   Number of data points ton be built
 * \param[out] d_cmpx  Complex numbers
 *
 */
template <typename TREAL, typename TIMAG, typename TCMPX>
  __global__ void cudautil_complexbuilder(const TREAL *d_real, const TIMAG *d_imag, TCMPX *d_cmpx, int ndata){
  // Maximum x-dimension of a grid of thread blocks is 2^31-1
  // Maximum x- or y-dimension of a block is 1024
  // So here we can cover (2^31-1)*1024 random numbers, which are 2^41-1024
  // should be big enough
  
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<ndata){

    scalar_typecast(d_real[idx], d_cmpx[idx].x);
    scalar_typecast(d_imag[idx], d_cmpx[idx].y);
  }
}

/*! \brief A class to build a complex vector with two real vectors
 *
 * \tparam TREAL Typename of the real part data
 * \tparam TIMAG Typename of the imag part data
 * \tparam TCMPX Typename of the complex data
 *
 * The class use kernel `cudautil_complexbuilder` to convert data type and build complex numbers. `cudautil_complexbuilder` uses `scalar_typecast` to convert data type. 
 * As of that, the allowed data type here is limited to following table (more types can be added later) 
 * 
 * TREAL/TIMAG    | TCMPX
 * -------|----
 * float  | cuComplex
 * float  | cuDoubleComplex
 * float  | half2
 * float  | int2
 * float  | short2
 * float  | int8_t ???
 * double | cuComplex
 * half   | cuComplex
 * int    | cuComplex
 * int16_t| cuComplex
 * int8_t | cuComplex
 *
 */
template <typename TREAL, typename TIMAG, typename TCMPX>
  class ComplexBuilder {
 public:
  TCMPX *data = NULL; ///< Complex data on device
  
  //! Constructor of ComplexBuilder class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata complex numbers
   * - build complex numbers with \p real and \p imag
   *
   * \see cudautil_complexbuilder, scalar_typecast
   *
   * \tparam TREAL Real part data type
   * \tparam TIMAG Imag part data type
   * \tparam TCMPX Complex data type
   * 
   * \param[in] real  Real part to build complex numbers
   * \param[in] imag  Imag part to build complex numbers
   * \param[in] ndata   Number of data points ton be converted
   * \param[in] nthread Number of threads per CUDA block to run `cudautil_complexbuilder` kernel
   *
   */
 ComplexBuilder(TREAL *real, TIMAG *imag, int ndata, int nthread )
   :ndata(ndata), nthread(nthread){

    // Sort out input data
    data_real = copy2device(real, ndata, type_real);
    data_imag = copy2device(imag, ndata, type_imag);

    // Create output buffer
    checkCudaErrors(cudaMallocManaged(&data, ndata*sizeof(TCMPX), cudaMemAttachGlobal));

    // Setup kernel size and run it 
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;
    
    cudautil_complexbuilder<<<nblock, nthread>>>(data_real, data_imag, data, ndata);
    getLastCudaError("Kernel execution failed [ cudautil_complexbuilder ]");

    // Free intermediate memory
    remove_device_copy(type_real, data_real);
    remove_device_copy(type_imag, data_imag);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of ComplexBuilder class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~ComplexBuilder(){
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
 private:
  TREAL *data_real = NULL;
  TIMAG *data_imag = NULL;

  enum cudaMemoryType type_real; ///< memory type
  enum cudaMemoryType type_imag; ///< memory type
  
  int ndata;
  int nthread;
  int nblock;
};

//! A template kernel to split complex numbers into its real and imag part
/*!
 * 
 * \see scalar_typecast
 *
 * \tparam TREAL Real part data type
 * \tparam TIMAG Imag part data type
 * \tparam TCMPX Complex data type
 * 
 * \param[in]  d_cmpx  Complex numbers 
 * \param[out] d_real  Real part of complex numbers
 * \param[out] d_imag  Imag part of complex numbers
 * \param[in]  ndata   Number of data points to be splitted
 *
 */
template <typename TCMPX, typename TREAL, typename TIMAG>
  __global__ void cudautil_complexsplitter(const TCMPX *d_cmpx, const TREAL *d_real, TIMAG *d_imag, int ndata){
  // Maximum x-dimension of a grid of thread blocks is 2^31-1
  // Maximum x- or y-dimension of a block is 1024
  // So here we can cover (2^31-1)*1024 random numbers, which are 2^41-1024
  // should be big enough
  
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx<ndata){

    scalar_typecast(d_cmpx[idx].x,   d_real[idx]);
    scalar_typecast(d_cmpx[idx].y, d_imag[idx]);
  }
}

/*! \brief A class to build a complex vector with two real vectors
 *
 * \tparam TREAL Typename of the real part data
 * \tparam TIMAG Typename of the imag part data
 * \tparam TCMPX Typename of the complex data
 *
 * The class use kernel `cudautil_complexbuilder` to convert data type and build complex numbers. `cudautil_complexbuilder` uses `scalar_typecast` to convert data type. 
 * As of that, the allowed data type here is limited to following table (more types can be added later) 
 * 
 * TREAL/TIMAG    | TCMPX
 * -------|----
 * float  | cuComplex
 * float  | cuDoubleComplex
 * float  | half2
 * float  | int2
 * float  | short2
 * float  | int8_t ???
 * double | cuComplex
 * half   | cuComplex
 * int    | cuComplex
 * int16_t| cuComplex
 * int8_t | cuComplex
 *
 */
template <typename TCMPX, typename TREAL, typename TIMAG>
  class ComplexSplitter {
 public:
  TREAL *real = NULL; ///< Real part on device
  TIMAG *imag = NULL; ///< Imag part on device
  
  //! Constructor of ComplexSplitter class.
  /*!
   * 
   * - initialise the class
   * - create device memory for \p ndata real and imag numbers
   * - split complex numbers to \p real and \p imag
   *
   * \see cudautil_complexbuilder, scalar_typecast
   *
   * \tparam TREAL Real part data type
   * \tparam TIMAG Imag part data type
   * \tparam TCMPX Complex data type
   * 
   * \param[in] cmpx  Complex numbers
   * \param[in] ndata   Number of data points ton be converted
   * \param[in] nthread Number of threads per CUDA block to run `cudautil_complexbuilder` kernel
   *
   */
  
 ComplexSplitter(TCMPX *cmpx, int ndata, int nthread)
   :ndata(ndata), nthread(nthread){

    // Sort out input buffer
    data = copy2device(cmpx, ndata, type);
    
    // Create managed memory for output
    checkCudaErrors(cudaMallocManaged(&real, ndata*sizeof(TREAL), cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&imag, ndata*sizeof(TIMAG), cudaMemAttachGlobal));

    // Setup kernel and run it 
    nblock = ndata/nthread;
    nblock = (nblock>1)?nblock:1;
    
    cudautil_complexsplitter<<<nblock, nthread>>>(data, real, imag, ndata);
    getLastCudaError("Kernel execution failed [ cudautil_complexsplitter ]");

    // Free intermediate memory
    remove_device_copy(type, data);    
    checkCudaErrors(cudaDeviceSynchronize());
  }
  
  //! Deconstructor of ComplexSplitter class.
  /*!
   * 
   * - free device memory at the class life end
   */
  ~ComplexSplitter(){
    checkCudaErrors(cudaFree(real));
    checkCudaErrors(cudaFree(imag));

    checkCudaErrors(cudaDeviceSynchronize());
  }
  
 private:
  TCMPX *data = NULL;
  
  enum cudaMemoryType type; ///< memory type
    
  int ndata;
  int nthread;
  int nblock;
};


#endif // CUDA_CONVERTORS_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
