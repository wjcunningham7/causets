#ifndef SUBROUTINES_GPU_H_
#define SUBROUTINES_GPU_H_

#include "CuResources.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

__host__ __device__ uint64_t vec2MatIdx(const int &N, const uint64_t &vecIdx);

__device__ void swap(uint64_t * const &edges, const unsigned int &i, const unsigned int &j);

#endif
