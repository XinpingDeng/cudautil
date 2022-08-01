// cuda_overload.h
//
// last-edit-by: <> 
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#pragma once

#ifndef CUDA_OVERLOAD_H
#define CUDA_OVERLOAD_H 1

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

/*! A function to overload make_cuComplex
 */
template <typename TREAL, typename TIMAG, typename TCMPX>
  __device__ static inline void make_cuComplex(const TREAL x, const TIMAG y, TCMPX &z){
  scalar_typecast(x, z.x);
  scalar_typecast(y, z.y);
}


#endif // CUDA_OVERLOAD_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
