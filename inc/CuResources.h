/////////////////////////////
//(C) Will Cunningham 2017 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef CU_RESOURCES_H_
#define CU_RESOURCES_H_

/*#ifdef __CUDACC_VER__
#undef __CUDACC_VER__
#define __CUDACC_VER__ 90000
#endif*/

#include <cassert>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

/*#ifdef CUDA_ENABLED
#include <cuda.h>
#include <drvapi_error_string.h>
#endif*/

#include <fastmath/stopwatch.h>
#include "config.h"
#include "Subroutines.h"

void printMemUsed(char const * chkPoint, size_t hostMem, size_t devMem, const int &rank);
void memoryCheckpoint(const size_t &hostMemUsed, size_t &maxHostMemUsed, const size_t &devMemUsed, size_t &maxDevMemUsed);

#endif
