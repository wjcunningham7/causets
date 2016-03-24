#ifndef SUBROUTINES_GPU_H_
#define SUBROUTINES_GPU_H_

#include "CuResources.h"
#include "Constants.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

__host__ __device__ uint64_t vec2MatIdx(const int &N, const uint64_t &vecIdx);

__global__ void BitonicSort(uint64_t *edges, int j, int k);

__device__ void swap(uint64_t * const &edges, const unsigned int &i, const unsigned int &j);

__global__ void Scan(int *input, int64_t *output, int *buf, int elements);

__global__ void PostScan(int64_t *input, int64_t *buf, int elements);

#endif
