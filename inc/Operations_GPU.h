/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef OPERATIONS_GPU_H_
#define OPERATIONS_GPU_H_

#include "Causet.h"
#include "CuResources.h"
#include "Subroutines_GPU.h"

#define POW2_GPU(x) ((x)*(x))

__device__ float X1_GPU(const float &theta1);

__device__ float X2_GPU(const float &theta1, const float &theta2);

__device__ float X3_GPU(const float &theta1, const float &theta2, const float &theta3);

__device__ float X4_GPU(const float &theta1, const float &theta2, const float &theta3);

__device__ float X_FLAT_GPU(const float &theta1, const float &theta2, const float &theta3);

__device__ float Y_FLAT_GPU(const float &theta1, const float &theta2, const float &theta3);

__device__ float Z_FLAT_GPU(const float &theta1, const float &theta2);

__device__ float sphProduct_GPU_v1(const float4 &sc0, const float4 &sc1);

__device__ float sphProduct_GPU_v2(const float4 &sc0, const float4 &sc1);

__device__ float flatProduct_GPU_v1(const float4 &sc0, const float4 &sc1);

__device__ float flatProduct_GPU_v2(const float4 &sc0, const float4 &sc1);

__device__ float flatProduct3_GPU(const float4 &sc0, const float4 &sc1);

__device__ float sphProduct3_GPU(const float4 &sc0, const float4 &sc1);

#endif
