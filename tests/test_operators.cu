#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "util.hpp"
#include "util.cuh"

#include <catch2/catch_test_macros.hpp>

using namespace std;

TEST_CASE("cuComplexTimesFloat", "cuComplexTimesFloat") {
  
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

TEST_CASE("floatTimescuComplex", "floatTimescuComplex") {
  
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

TEST_CASE("cuComplexDividesFloat", "cuComplexDividesFloat") {
  
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

TEST_CASE("cuComplexDividesEqualFloat", "cuComplexDividesEqualFloat") {
  
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

TEST_CASE("cuComplexPlusEqualcuComplex", "cuComplexPlusEqualcuComplex") {
  
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

TEST_CASE("cuComplexMinusEqualcuComplex", "cuComplexMinusEqualcuComplex") {
  
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

