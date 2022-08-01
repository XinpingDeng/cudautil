// test_utilities.h
//
// last-edit-by: <> 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once

#ifndef TEST_UTILITIES_H
#define TEST_UTILITIES_H 1

#define __STDC_FORMAT_MACROS

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

  inline int seconds2dhms(uint64_t seconds, char *dhms){

    uint64_t n = seconds;
  
    int days = n / (24 * 3600);
  
    n = n % (24 * 3600);
    int hours = n / 3600;
  
    n %= 3600;
    int minutes = n / 60 ;
  
    n %= 60;

    sprintf(dhms, "%d:%02d:%02d:%02d", days, hours, minutes, (int)n);
  
    return EXIT_SUCCESS;
  }
  //int seconds2dhms(uint64_t seconds, char *dhms);

  // The following copied from https://stackoverflow.com/questions/41390824/%C2%B5s-precision-wait-in-c-for-linux-that-does-not-put-program-to-sleep
# define tscmp(a, b, CMP)			\
  (((a)->tv_sec == (b)->tv_sec) ?		\
   ((a)->tv_nsec CMP (b)->tv_nsec) :		\
   ((a)->tv_sec CMP (b)->tv_sec))
# define tsadd(a, b, result)				\
  do {							\
    (result)->tv_sec = (a)->tv_sec + (b)->tv_sec;	\
    (result)->tv_nsec = (a)->tv_nsec + (b)->tv_nsec;	\
    if ((result)->tv_nsec >= 1000000000)		\
      {							\
	++(result)->tv_sec;				\
	(result)->tv_nsec -= 1000000000;		\
      }							\
  } while (0)
# define tssub(a, b, result)				\
  do {							\
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;	\
    (result)->tv_nsec = (a)->tv_nsec - (b)->tv_nsec;	\
    if ((result)->tv_nsec < 0) {			\
      --(result)->tv_sec;				\
      (result)->tv_nsec += 1000000000;			\
    }							\
  } while (0)

#ifdef __cplusplus
}
#endif




#endif // TEST_UTILITIES_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
