#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "shared_utils.h"

bool approximates(const std::complex<float> &a, const std::complex<float> &b, unsigned nsamples){
  float absolute = abs(a - b), relative = abs(a / b);

  return (absolute < (.0001 * nsamples)) || (relative > .999 && relative < 1.001);
}
