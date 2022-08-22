#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "dada_utils.h"

int dada_verify_block_size(int nbytes_expected, ipcbuf_t *block){

  int nbytes_actual = ipcbuf_get_bufsz(block);
  
  if (nbytes_expected != nbytes_actual){
    fprintf(stderr, "input buffer block size mismatch, "
  	    "expected %d bytes, but actual %d bytes"
  	    "which happens at \"%s\", line [%d], has to abort.\n",
  	    nbytes_expected, nbytes_actual,
  	    __FILE__, __LINE__);	
    
    return EXIT_FAILURE;
  }

  fprintf(stdout, "We have input buffer block size checked\n");
  fprintf(stdout, "\n");

  return EXIT_SUCCESS;
}

dada_hdu_t* dada_setup_hdu(key_t key, int read, multilog_t* log){

  dada_hdu_t *hdu = dada_hdu_create(log);
  dada_hdu_set_key(hdu, key);
  if(dada_hdu_connect(hdu) < 0){ 
    fprintf(stderr, "Can not connect to hdu with key %x, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);    
  }

  if(read){    
    if(dada_hdu_lock_read(hdu) < 0) {
      fprintf(stderr, "Error locking input HDU with key %x, \n"
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      key, __FILE__, __LINE__);
      
      exit(EXIT_FAILURE);
    }
    fprintf(stdout, "We have input HDU with key %x locked\n", key);
  }
  else{
    if(dada_hdu_lock_write(hdu) < 0) {
      fprintf(stderr, "Error locking output HDU with key %x, \n"
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      key, __FILE__, __LINE__);
      
      exit(EXIT_FAILURE);
    }
    fprintf(stdout, "We have output HDU with key %x locked\n", key);    
  }
  
  return hdu;
}

ipcbuf_t *dada_get_data_block(dada_hdu_t *hdu){

  key_t key=hdu->data_block_key;
  
  ipcbuf_t *data_block = (ipcbuf_t *)(hdu->data_block);
  
  if(data_block == NULL){
      fprintf(stderr, "Error getting data block from HDU with key %x, \n"
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }

  fprintf(stdout, "We get data block from HDU with key %x locked\n", key);
  
  return data_block;
}

ipcbuf_t *dada_get_header_block(dada_hdu_t *hdu){

  key_t key=hdu->header_block_key;
  
  ipcbuf_t *header_block = (ipcbuf_t *)(hdu->header_block);
  
  if(header_block == NULL){
      fprintf(stderr, "Error getting header block from HDU with key %x, \n"
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      key, __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }
  
  fprintf(stdout, "We get header block from HDU with key %x locked\n", key);

  return header_block;
}
