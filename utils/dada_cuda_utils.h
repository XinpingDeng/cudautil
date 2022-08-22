#ifndef _DADA_CUDA_UTILS_H
#define _DADA_CUDA_UTILS_H

#include <stdlib.h>

#include "ipcio.h"
#include "futils.h"
#include "ipcbuf.h"
#include "dada_def.h"
#include "ascii_header.h"
#include "dada_hdu.h"
#include "multilog.h"

#include "dada_def.h"

#include "dada_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

  int dada_dbregister(dada_hdu_t *hdu);
  
#ifdef __cplusplus
}
#endif

#endif
