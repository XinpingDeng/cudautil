#ifndef _CUDAUTIL_H
#define _CUDAUTIL_H

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <curand.h>

/*! \brief A function to check CUDA global memory.
 *
 * This function prints out total and available CUDA global memory in MBytes. 
 * 
 */
int printCudaMemInfo();

int print_memory_info();

//**************************
//***************************
//***************************
// How we round double to int is important to calculate sampleshift in calculatedelayandphase function and its cuda version? Using different cast function here gives us different shifts.
// Using the second one gives identical result from C and CUDA, but that is what we want or not? I mean cast double to int as we normally use (int) is the right result we are looking for?
//***************************
//***************************
//***************************
//__device__ static inline int cuDouble2Int(double a){return __double2int_rn(a);} // This is the one used in Chris's code
__device__ static inline int cuDouble2Int(double a){return __double2int_rz(a);}

__device__ static inline int   cuFloat2Int(float a) {return __float2int_rz(a);}
__device__ static inline half  cuFloat2Half(float a){return __float2half(a);}
__device__ static inline float cuHalf2Float(half a) {return __half2float(a);}

// We need more type case overload functions here
__device__ static inline void scalar_typecast(const float a, float   &b) { b = a;}

__device__ static inline void scalar_typecast(const float a, double  &b) { b = a;}
__device__ static inline void scalar_typecast(const float a, half    &b) { b = cuFloat2Half(a);}
__device__ static inline void scalar_typecast(const float a, int     &b) { b = cuFloat2Int(a);}
__device__ static inline void scalar_typecast(const float a, int16_t &b) { b = cuFloat2Int(a);}
__device__ static inline void scalar_typecast(const float a, int8_t  &b) { b = cuFloat2Int(a);}

__device__ static inline void scalar_typecast(const double a,  float &b) { b = a;}
__device__ static inline void scalar_typecast(const half a,    float &b) { b = cuHalf2Float(a);}
__device__ static inline void scalar_typecast(const int a,     float &b) { b = a;}
__device__ static inline void scalar_typecast(const int16_t a, float &b) { b = a;}
__device__ static inline void scalar_typecast(const int8_t a,  float &b) { b = a;}
__device__ static inline void scalar_typecast(const unsigned a,  float &b) { b = a;}

template <typename TIN, typename TOUT>
__device__ static inline void complex_typecast(const TIN a, TOUT &b){
  
  scalar_typecast(a.x, b.x);
  scalar_typecast(a.y, b.y);
}

template <typename TREAL, typename TIMAG, typename TCMPX>
__device__ static inline void cuMakeComplex(const TREAL x, const TIMAG y, TCMPX &z){

  scalar_typecast(x, z.x);
  scalar_typecast(y, z.y);
}

template <typename TMIN, typename TSUB, typename TRES>
__device__ static inline void scalar_subtract(const TMIN minuend, const TSUB subtrahend, TRES &result) {
  TRES casted_minuend;
  TRES casted_subtrahend;
  
  scalar_typecast(minuend,    casted_minuend);
  scalar_typecast(subtrahend, casted_subtrahend);
  
  result = casted_minuend - casted_subtrahend;
}

#endif
