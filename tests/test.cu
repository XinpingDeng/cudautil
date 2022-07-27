#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "cudautil.h"

#include <iostream>

int main(int argc, char *argv[]){

  print_cuda_memory_info();

  float b = 3.0;
  cuComplex a = {10.0, 10.0};

  cuComplex c = a/b;

  std::cout << c.x << "\t" 
	    << c.y << std::endl;
  
  return EXIT_SUCCESS;
}    
