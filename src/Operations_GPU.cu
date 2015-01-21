#include "Operations_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//BEGIN COMPACT EQUATIONS (Completed)

__device__ float X1_SPH_GPU(const float &phi)
{
	return cosf(phi);
}

__device__ float X2_SPH_GPU(const float &phi, const float &chi)
{
	return sinf(phi) * cosf(chi);
}

__device__ float X3_SPH_GPU(const float &phi, const float &chi, const float &theta)
{
	return sinf(phi) * sinf(chi) * cosf(theta);
}

__device__ float X4_SPH_GPU(const float &phi, const float &chi, const float &theta)
{
	return sinf(phi) * sinf(chi) * sinf(theta);
}

__device__ float X_FLAT_GPU(const float &phi, const float &chi, const float &theta)
{
	return chi * cosf(theta) * sinf(phi);
}

__device__ float Y_FLAT_GPU(const float &phi, const float &chi, const float &theta)
{
	return chi * sinf(theta) * sinf(phi);
}

__device__ float Z_FLAT_GPU(const float &phi, const float &chi)
{
	return chi * cosf(phi);
}

__device__ float sphProduct_GPU_v1(const float4 &sc0, const float4 &sc1)
{
	return X1_SPH_GPU(sc0.y) * X1_SPH_GPU(sc1.y) +
	       X2_SPH_GPU(sc0.y, sc0.z) * X2_SPH_GPU(sc1.y, sc1.z) +
	       X3_SPH_GPU(sc0.y, sc0.z, sc0.x) * X3_SPH_GPU(sc1.y, sc1.z, sc1.x) +
	       X4_SPH_GPU(sc0.y, sc0.z, sc0.x) * X4_SPH_GPU(sc1.y, sc1.z, sc1.x);
}

__device__ float sphProduct_GPU_v2(const float4 &sc0, const float4 &sc1)
{
	return cosf(sc0.y) * cosf(sc1.y) +
	       sinf(sc0.y) * sinf(sc1.y) * (cosf(sc0.z) * cosf(sc1.z) +
	       sinf(sc0.z) * sinf(sc1.z) * cosf(sc0.x - sc1.x));
}

__device__ float flatProduct_GPU_v1(const float4 &sc0, const float4 &sc1)
{
	return POW2_GPU(X_FLAT_GPU(sc0.y, sc0.z, sc0.x) - X_FLAT_GPU(sc1.y, sc1.z, sc1.x)) +
	       POW2_GPU(Y_FLAT_GPU(sc0.y, sc0.z, sc0.x) - Y_FLAT_GPU(sc1.y, sc1.z, sc1.x)) +
	       POW2_GPU(Z_FLAT_GPU(sc0.y, sc0.z) - Z_FLAT_GPU(sc1.y, sc1.z));
}

__device__ float flatProduct_GPU_v2(const float4 &sc0, const float4 &sc1)
{
	return POW2_GPU(sc0.z) + POW2_GPU(sc1.z) -
	       2.0f * sc0.z * sc1.z * (cosf(sc0.y) * cosf(sc1.y) +
	       sinf(sc0.y) * sinf(sc1.y) * cosf(sc0.x - sc1.x));
}

__device__ float POW2_GPU(const float &x)
{
	return x * x;
}

//END COMPACT EQUATIONS
