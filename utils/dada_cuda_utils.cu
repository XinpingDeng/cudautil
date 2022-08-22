#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "dada_cuda_utils.h"

int dada_dbregister(dada_hdu_t *hdu){
  
  key_t key = hdu->data_block_key;
  
  if(dada_dbregister(hdu) < 0){
    fprintf(stderr, "Error dbregistering HDU with key %x, \n"
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "We have HDU with key %x dbregistered\n", key);

  return EXIT_SUCCESS;
}

