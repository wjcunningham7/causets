#include "Operations_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
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
