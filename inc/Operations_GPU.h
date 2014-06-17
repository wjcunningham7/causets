#ifndef OPERATIONS_GPU_H_
#define OPERATIONS_GPU_H_

#include "Causet.h"
#include "CuResources.h"
#include "Subroutines_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

__device__ float X1_GPU(const float &phi);

__device__ float X2_GPU(const float &phi, const float &chi);

__device__ float X3_GPU(const float &phi, const float &chi, const float &theta);

__device__ float X4_GPU(const float &phi, const float &chi, const float &theta);

#endif
