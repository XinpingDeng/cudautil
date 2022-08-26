#ifndef _UTILS_H
#define _UTILS_H

#pragma once

#include <complex>
bool approximates(const std::complex<float> &a, const std::complex<float> &b, unsigned nsamples){
  float absolute = abs(a - b), relative = abs(a / b);

  return (absolute < (.0001 * nsamples)) || (relative > .999 && relative < 1.001);
}

#endif // UTILS_H
