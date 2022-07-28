#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.hpp"
#include "util.cuh"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("cuComplex_divide", "cuComplex_divide") {

  print_cuda_memory_info();

  float b = 3.0;
  cuComplex a = {10.0, 10.0};

  cuComplex c = a/b;

  std::cout << a << std::endl;
}    
