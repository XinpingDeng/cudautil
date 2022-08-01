// cuda_timer.h
//
// last-edit-by: <> 
//
// Description:
//
//////////////////////////////////////////////////////////////////////

#pragma once

#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H 1

#include <cuda.h>

#include <helper_cuda.h>

#define CUDA_STARTTIME(x)  cudaEventRecord(x ## _start, 0);
#define CUDA_STOPTIME(x) {					\
    float dtime;						\
    cudaEventRecord(x ## _stop, 0);				\
    cudaEventSynchronize(x ## _stop);				\
    cudaEventElapsedTime(&dtime, x ## _start, x ## _stop);	\
    x ## time += dtime; }



#endif // CUDA_TIMER_H
//////////////////////////////////////////////////////////////////////
// $Log:$
//
