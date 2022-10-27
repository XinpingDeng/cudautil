#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "utils/cuda_utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

using namespace std;

TEST_CASE("cuComplexTimesFloat") {
  
  cuComplex a = {10.0, 5.0};
  float b = 3.0;
  cuComplex c = a*b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a*b is " << c << endl;
  
  CHECK(a.x*b == c.x);
  CHECK(a.y*b == c.y);
}    

TEST_CASE("floatTimescuComplex") {
  
  cuComplex a = {10.0, 5.0};
  float b = 3.0;
  cuComplex c = b*a;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "b*a is " << c << endl;
  
  CHECK(b*a.x == c.x);
  CHECK(b*a.y == c.y);
}    

TEST_CASE("cuComplexDividesFloat") {
  
  cuComplex a = {10.0, 5.0};
  float b = 3.0;
  cuComplex c = a/b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a/b is " << c << endl;
  
  CHECK(a.x/b == c.x);
  CHECK(a.y/b == c.y);
}    

TEST_CASE("cuComplexDividesEqualFloat") {
  
  cuComplex a = {10.0, 5.0};
  float b = 3.0;
  cuComplex c = a;
  
  a/=b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a/=b is " << a << endl;
  
  CHECK(a.x == c.x/b);
  CHECK(a.y == c.y/b);
}

TEST_CASE("cuComplexPlusEqualcuComplex") {
  
  cuComplex a = {10.0, 5.0};
  cuComplex b = {3.0, 1.0};
  cuComplex c = a;
  
  a+=b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a+=b is " << a << endl;
  
  CHECK(a.x == c.x+b.x);
  CHECK(a.y == c.y+b.y);
}    

TEST_CASE("cuComplexMinusEqualcuComplex") {
  
  cuComplex a = {10.0, 5.0};
  cuComplex b = {3.0, 1.0};
  cuComplex c = a;
  
  a-=b;

  cout << endl;
  cout << a << endl;
  cout << b << endl;
  cout << "a-=b is " << a << endl;
  
  CHECK(a.x == c.x-b.x);
  CHECK(a.y == c.y-b.y);
}    

