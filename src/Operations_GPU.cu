#include "Operations_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

__device__ float X1_GPU(const float &phi)
{
	return cosf(phi);
}

__device__ float X2_GPU(const float &phi, const float &chi)
{
	return sinf(phi) * cosf(chi);
}

__device__ float X3_GPU(const float &phi, const float &chi, const float &theta)
{
	return sinf(phi) * sinf(chi) * cosf(theta);
}

__device__ float X4_GPU(const float &phi, const float &chi, const float &theta)
{
	return sinf(phi) * sinf(chi) * sinf(theta);
}

__device__ float sphProduct_GPU(const float4 &sc0, const float4 &sc1)
{
	return X1_GPU(sc0.y) * X1_GPU(sc1.y) +
	       X2_GPU(sc0.y, sc0.z) * X2_GPU(sc1.y, sc1.z) +
	       X3_GPU(sc0.y, sc0.z, sc0.x) * X3_GPU(sc1.y, sc1.z, sc1.x) +
	       X4_GPU(sc0.y, sc0.z, sc0.x) * X4_GPU(sc1.y, sc1.z, sc1.x);
}
