#ifndef OPERATIONS_GPU_H_
#define OPERATIONS_GPU_H_

#include "Causet.h"
#include "CuResources.h"
#include "Subroutines_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

__device__ float X1_GPU(const float &phi);

__device__ float X2_GPU(const float &phi, const float &chi);

__device__ float X3_GPU(const float &phi, const float &chi, const float &theta);

__device__ float X4_GPU(const float &phi, const float &chi, const float &theta);

__device__ float X_FLAT_GPU(const float &phi, const float &chi, const float &theta);

__device__ float Y_FLAT_GPU(const float &phi, const float &chi, const float &theta);

__device__ float Z_FLAT_GPU(const float &phi, const float &chi);

__device__ float sphProduct_GPU(const float4 &sc0, const float4 &sc1);

__device__ float flatProduct_GPU(const float4 &sc0, const float4 &sc1);

__device__ float POW2_GPU(const float &x);

#endif
