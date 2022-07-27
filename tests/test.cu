#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "cudautil.h"

#include <catch2/catch_test_macros.hpp>

#include <iostream>

TEST_CASE("cuComplex_divide", "cuComplex_divide") {

  print_cuda_memory_info();

  float b = 3.0;
  cuComplex a = {10.0, 10.0};

  cuComplex c = a/b;
}    
