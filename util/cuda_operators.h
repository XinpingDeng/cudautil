// cuda_operator.h
//
// last-edit-by: <> 
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#pragma once

#ifndef CUDA_OPERATOR_H
#define CUDA_OPERATOR_H 1


#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

/*! Overload * operator to multiple a cuComplex with a float for device and host code
 *
 * \param[in] a Input cuComplex number
 * \param[in] b Input float number
 * \returns   \a a * \a b
 *
 */
__device__ __host__ static inline cuComplex operator*(cuComplex a, float b) { return make_cuComplex(a.x*b, a.y*b);}

/*! Overload * operator to multiple a float with a cuComplex for device and host code
 *
 * \param[in] a Input float number
 * \param[in] b Input cuComplex number
 * \returns   \a a * \a b
 *
 */
__device__ __host__ static inline cuComplex operator*(float a, cuComplex b) { return make_cuComplex(b.x*a, b.y*a);}

/*! Overload / operator to divide a cuComplex with a float for device and host code
 *
 * \param[in, out] a A cuComplex number which will be divided by float \a b 
 * \param[in]      b Input float number
 * \returns        \a a / \a b
 *
 */
__device__ __host__ static inline cuComplex operator/(cuComplex a, float b) { return make_cuComplex(a.x/b, a.y/b);}

/*! Overload /= operator to divide a cuComplex with a float before it is accumulated to itself for device and host code
 *
 * \param[in, out] a A cuComplex number which will be divided by float \a b and accumulated to
 * \param[in]      b Input float number
 *
 */
__device__ __host__ static inline void operator/=(cuComplex &a, float b)     { a.x/=b;   a.y/=b;}

/*! Overload /= operator to plus a cuComplex with a cuComplex before it is accumulated to itself for device and host code
 *
 * \param[in, out] a A cuComplex number which will be added by cuComplex \a b and accumulated to
 * \param[in]      b Input cuComplex number
 *
 */
__device__ __host__ static inline void operator+=(cuComplex &a, cuComplex b) { a.x+=b.x; a.y+=b.y;}

/*! Overload /= operator to minus a cuComplex with a cuComplex before it is accumulated to itself for device and host code
 *
 * \param[in, out] a A cuComplex number which will be minused by cuComplex \a b and accumulated to
 * \param[in]      b Input cuComplex number
 *
 */
__device__ __host__ static inline void operator-=(cuComplex &a, cuComplex b) { a.x-=b.x; a.y-=b.y;}

/*! Overload cout with cuda complex data type
*/
#include <iostream>
static inline std::ostream& operator<<(std::ostream& os, const cuComplex& data){
  os << data.x << ' ' << data.y << ' ';
  return os;
}

#endif // CUDA_OPERATOR_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
