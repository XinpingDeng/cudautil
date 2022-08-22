#ifndef __DADA_HEADER_H
#define __DADA_HEADER_H

#define DADA_STRLEN 1024

#ifdef __cplusplus
extern "C" {
#endif

#include "inttypes.h"

  typedef struct dada_header_t{
    double tsamp;
    double mjd_start;
    double bw;
    char utc_start[DADA_STRLEN];
    int nchan;
    int npkt;
    int pkt_nsamp;
    int nchan_fine;
    int naverage;
    double pkt_tsamp;
    int npol;
    int nant;
    int nbit;
    uint64_t totalsamples;
    double period;
  }dada_header_t;

  int read_dada_header(const char *dada_header_buffer, dada_header_t *dada_header);

  int write_dada_header(const dada_header_t dada_header, char *dada_header_buffer);

  int read_dada_header_from_file(const char *dada_header_file_name, dada_header_t *dada_header);

  int write_dada_header_to_file(const dada_header_t dada_header,const char *dada_header_file_name);

#ifdef __cplusplus
}
#endif

#endif
