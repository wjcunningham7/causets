#include "Operations_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

__device__ float X1_SPH_GPU(const float &theta1)
{
	return cosf(theta1);
}

__device__ float X2_SPH_GPU(const float &theta1, const float &theta2)
{
	return sinf(theta1) * cosf(theta2);
}

__device__ float X3_SPH_GPU(const float &theta1, const float &theta2, const float &theta3)
{
	return sinf(theta1) * sinf(theta2) * cosf(theta3);
}

__device__ float X4_SPH_GPU(const float &theta1, const float &theta2, const float &theta3)
{
	return sinf(theta1) * sinf(theta2) * sinf(theta3);
}

__device__ float X_FLAT_GPU(const float &theta1, const float &theta2, const float &theta3)
{
	return theta1 * sinf(theta2) * cosf(theta3);
}

__device__ float Y_FLAT_GPU(const float &theta1, const float &theta2, const float &theta3)
{
	return theta1 * sinf(theta2) * sinf(theta3);
}

__device__ float Z_FLAT_GPU(const float &theta1, const float &theta2)
{
	return theta1 * cosf(theta2);
}

__device__ float sphProduct_GPU_v1(const float4 &sc0, const float4 &sc1)
{
	return X1_SPH_GPU(sc0.x) * X1_SPH_GPU(sc1.x) +
	       X2_SPH_GPU(sc0.x, sc0.y) * X2_SPH_GPU(sc1.x, sc1.y) +
	       X3_SPH_GPU(sc0.x, sc0.y, sc0.z) * X3_SPH_GPU(sc1.x, sc1.y, sc1.z) +
	       X4_SPH_GPU(sc0.x, sc0.y, sc0.z) * X4_SPH_GPU(sc1.x, sc1.y, sc1.z);
}

__device__ float sphProduct_GPU_v2(const float4 &sc0, const float4 &sc1)
{
	return cosf(sc0.x) * cosf(sc1.x) +
	       sinf(sc0.x) * sinf(sc1.x) * (cosf(sc0.y) * cosf(sc1.y) +
	       sinf(sc0.y) * sinf(sc1.y) * cosf(sc0.z - sc1.z));
}

//NOTE: v1 and v2 for flatProduct can lead to very slightly different results (not entirely sure why)
//      v1 matches for CPU/GPU and v2 leads to the erroneous anomaly

__device__ float flatProduct_GPU_v1(const float4 &sc0, const float4 &sc1)
{
	return POW2_GPU(X_FLAT_GPU(sc0.x, sc0.y, sc0.z) - X_FLAT_GPU(sc1.x, sc1.y, sc1.z)) +
	       POW2_GPU(Y_FLAT_GPU(sc0.x, sc0.y, sc0.z) - Y_FLAT_GPU(sc1.x, sc1.y, sc1.z)) +
	       POW2_GPU(Z_FLAT_GPU(sc0.x, sc0.y) - Z_FLAT_GPU(sc1.x, sc1.y));
}

__device__ float flatProduct_GPU_v2(const float4 &sc0, const float4 &sc1)
{
	return POW2_GPU(sc0.x) + POW2_GPU(sc1.x) -
	       2.0f * sc0.x * sc1.x * (cosf(sc0.y) * cosf(sc1.y) +
	       sinf(sc0.y) * sinf(sc1.y) * cosf(sc0.z - sc1.z));
}

__device__ float POW2_GPU(const float &x)
{
	return x * x;
}
