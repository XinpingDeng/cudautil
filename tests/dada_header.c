#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "futils.h"
#include "dada_def.h"
#include "ascii_header.h"

#include "dada_header.h"

#include <stdlib.h>
#include <stdio.h>

#include <string.h>

int read_dada_header_from_file(const char *dada_header_file_name, dada_header_t *dada_header){

  char *dada_header_buffer = (char *)malloc(DADA_DEFAULT_HEADER_SIZE);
  memset(dada_header_buffer, 0, DADA_DEFAULT_HEADER_SIZE);

  fileread(dada_header_file_name, dada_header_buffer, DADA_DEFAULT_HEADER_SIZE);
  read_dada_header(dada_header_buffer, dada_header);

  free(dada_header_buffer);

  return EXIT_SUCCESS;
}

int read_dada_header(const char *dada_header_buffer, dada_header_t *dada_header){

  if (ascii_header_get(dada_header_buffer, "TSAMP", "%lf", &dada_header->tsamp) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "MJD_START", "%lf", &dada_header->mjd_start) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "BW", "%lf", &dada_header->bw) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting BW, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "UTC_START", "%s", &dada_header->utc_start) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting UTC_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NCHAN", "%d", &dada_header->nchan) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NPKT", "%d", &dada_header->npkt) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NPKT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_NSAMP", "%d", &dada_header->pkt_nsamp) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_NSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NCHAN_FINE", "%d", &dada_header->nchan_fine) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NCHAN_FINE, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NAVERAGE", "%d", &dada_header->naverage) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NAVERAGE, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PKT_TSAMP", "%lf", &dada_header->pkt_tsamp) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PKT_TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NPOL", "%d", &dada_header->npol) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NPOL, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NANT", "%d", &dada_header->nant) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NANT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "NBIT", "%d", &dada_header->nbit) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting NBIT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "TOTALSAMPLES", "%" PRIu64 "", &dada_header->totalsamples) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting TOTALSAMPLES, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_get(dada_header_buffer, "PERIOD", "%lf", &dada_header->period) < 0)  {
    fprintf(stderr, "WRITE_DADA_HEADER_ERROR: Error getting PERIOD, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

int write_dada_header_to_file(const dada_header_t dada_header, const char *dada_header_file_name){

  FILE *fp = fopen(dada_header_file_name, "w");
  char *dada_header_buffer = (char *)malloc(DADA_DEFAULT_HEADER_SIZE);
  memset(dada_header_buffer, 0, DADA_DEFAULT_HEADER_SIZE);

  sprintf(dada_header_buffer, "HDR_VERSION  1.0\nHDR_SIZE     4096\n");
  write_dada_header(dada_header, dada_header_buffer);
  fprintf(fp, "%s\n", dada_header_buffer);

  free(dada_header_buffer);
  fclose(fp);

  return EXIT_SUCCESS;
}

int write_dada_header(const dada_header_t dada_header, char *dada_header_buffer){

  if (ascii_header_set(dada_header_buffer, "TSAMP", "%f", dada_header.tsamp) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "MJD_START", "%f", dada_header.mjd_start) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting MJD_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "BW", "%f", dada_header.bw) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting BW, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "UTC_START", "%s", dada_header.utc_start) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting UTC_START, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NCHAN", "%d", dada_header.nchan) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NCHAN, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NPKT", "%d", dada_header.npkt) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NPKT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_NSAMP", "%d", dada_header.pkt_nsamp) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_NSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NCHAN_FINE", "%d", dada_header.nchan_fine) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NCHAN_FINE, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NAVERAGE", "%d", dada_header.naverage) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NAVERAGE, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PKT_TSAMP", "%f", dada_header.pkt_tsamp) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PKT_TSAMP, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NPOL", "%d", dada_header.npol) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NPOL, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NANT", "%d", dada_header.nant) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NANT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "NBIT", "%d", dada_header.nbit) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting NBIT, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "TOTALSAMPLES", "%" PRIu64 "", dada_header.totalsamples) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting TOTALSAMPLES, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (ascii_header_set(dada_header_buffer, "PERIOD", "%f", dada_header.period) < 0)  {
    fprintf(stderr, "READ_DADA_HEADER_ERROR: Error setting PERIOD, "
            "which happens at %s, line [%d].\n",
            __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
