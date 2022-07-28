#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "util.hpp"
#include "util.cuh"

#include <catch2/catch_test_macros.hpp>

using namespace std;

TEST_CASE("cuComplex_times_float", "cuComplex_times_float") {
  
  cuComplex a = {10.0, 5.0};
  float b = 3.0;
  cuComplex c = a*b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a*b is " << c << endl;
  
  REQUIRE(a.x*b == c.x);
  REQUIRE(a.y*b == c.y);
}    

TEST_CASE("float_times_cuComplex", "float_times_cuComplex") {
  
  cuComplex a = {10.0, 5.0};
  float b = 3.0;
  cuComplex c = b*a;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "b*a is " << c << endl;
  
  REQUIRE(b*a.x == c.x);
  REQUIRE(b*a.y == c.y);
}    

TEST_CASE("cuComplex_divides_float", "cuComplex_divides_float") {
  
  cuComplex a = {10.0, 5.0};
  float b = 3.0;
  cuComplex c = a/b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a/b is " << c << endl;
  
  REQUIRE(a.x/b == c.x);
  REQUIRE(a.y/b == c.y);
}    

TEST_CASE("cuComplex_divides_equal_float", "cuComplex_divides_equal_float") {
  
  cuComplex a = {10.0, 5.0};
  float b = 3.0;
  cuComplex c = a;
  
  a/=b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a/=b is " << a << endl;
  
  REQUIRE(a.x == c.x/b);
  REQUIRE(a.y == c.y/b);
}

TEST_CASE("cuComplex_plus_equal_cuComplex", "cuComplex_plus_equal_cuComplex") {
  
  cuComplex a = {10.0, 5.0};
  cuComplex b = {3.0, 1.0};
  cuComplex c = a;
  
  a+=b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a+=b is " << a << endl;
  
  REQUIRE(a.x == c.x+b.x);
  REQUIRE(a.y == c.y+b.y);
}    

TEST_CASE("cuComplex_minus_equal_cuComplex", "cuComplex_minus_equal_cuComplex") {
  
  cuComplex a = {10.0, 5.0};
  cuComplex b = {3.0, 1.0};
  cuComplex c = a;
  
  a-=b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a-=b is " << a << endl;
  
  REQUIRE(a.x == c.x-b.x);
  REQUIRE(a.y == c.y-b.y);
}    

