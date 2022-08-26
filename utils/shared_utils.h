#ifndef _SHARED_UTILS_H
#define _SHARED_UTILS_H

#pragma once

#include <complex>

bool approximates(const std::complex<float> &a, const std::complex<float> &b, unsigned nsamples);

// Dichotomy way of accumulate an array on CPU and return the result
template <typename T>
static T Sum2(T *a, int lo, int hi){
  // See here https://www.cnblogs.com/dx5800/p/13194664.html

  if (lo==hi){
    return a[lo];
  }
  
  int mi = (lo + hi) / 2;

  return Sum2(a, lo, mi) + Sum2(a, mi + 1, hi);
}

#endif // SHARED_UTILS_H
