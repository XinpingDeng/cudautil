// cuda_typecast.h
//
// last-edit-by: <> 
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#pragma once

#ifndef CUDA_TYPECAST_H
#define CUDA_TYPECAST_H 1

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CUDAUTIL_FLOAT2HALF __float2half
#define CUDAUTIL_FLOAT2INT  __float2int_rz
#define CUDAUTIL_FLOAT2UINT __float2uint_rz
#define CUDAUTIL_HALF2FLOAT __half2float
#define CUDAUTIL_DOUBLE2INT __double2int_rz

// We need more type case overload functions here
// The following convert float to other types
__device__ static inline void scalar_typecast(const float a, double   &b) { b = a;}
__device__ static inline void scalar_typecast(const float a, float    &b) { b = a;}
__device__ static inline void scalar_typecast(const float a, half     &b) { b = CUDAUTIL_FLOAT2HALF(a);}
__device__ static inline void scalar_typecast(const float a, int      &b) { b = CUDAUTIL_FLOAT2INT(a);}
__device__ static inline void scalar_typecast(const float a, int16_t  &b) { b = CUDAUTIL_FLOAT2INT(a);}
__device__ static inline void scalar_typecast(const float a, int8_t   &b) { b = CUDAUTIL_FLOAT2INT(a);}
__device__ static inline void scalar_typecast(const float a, unsigned &b) { b = CUDAUTIL_FLOAT2UINT(a);}

// The following convert other types to float
__device__ static inline void scalar_typecast(const double a,   float &b) { b = a;}
__device__ static inline void scalar_typecast(const half a,     float &b) { b = CUDAUTIL_HALF2FLOAT(a);}
__device__ static inline void scalar_typecast(const int a,      float &b) { b = a;}
__device__ static inline void scalar_typecast(const int16_t a,  float &b) { b = a;}
__device__ static inline void scalar_typecast(const int8_t a,   float &b) { b = a;}
__device__ static inline void scalar_typecast(const unsigned a, float &b) { b = a;}


#endif // CUDA_TYPECAST_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//

