#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "util.h"
#include "cudautil.h"

int main(int argc, char *argv[]){

  print_cuda_memory_info();
  
  return EXIT_SUCCESS;
}    
