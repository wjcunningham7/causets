#ifndef FAST_MATH_CPP_
#define FAST_MATH_CPP_

#include <boost/math/special_functions/gamma.hpp>

#include "fastapprox.h"
#include "FastMath.h"

//Approximation of x^2
float POW2(float x, int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1 || m == 2 || m == 3);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = powf(x, 2.0);
	else if (m == 1)
		y = x * x;
	else if (m == 2)
		//Defined in "fastapprox.h"
		y = fastpow(x, 2.0);
	else if (m == 3)
		//Defined in "fastapprox.h"
		y = fasterpow(x, 3.0);

	return y;
}

//Approximation of x^3
float POW3(float x, int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1 || m == 2 || m == 3);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = powf(x, 3.0);
	else if (m == 1)
		y = x * x * x;
	else if (m == 2)
		//Defined in "fastapprox.h"
		y = fastpow(x, 3.0);
	else if (m == 3)
		//Defined in "fastapprox.h"
		y = fasterpow(x, 3.0);

	return y;
}

//Approximation of x^p
float POW(float x, float p, int m)
{
	assert (m == 0 || m == 1 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = powf(x, p);
	else if (m == 1)
		//Defined in "fastapprox.h"
		y = fastpow(x, p);
	else if (m == 2)
		//Defined in "fastapprox.h"
		y = fasterpow(x, p);

	return y;
}

//Approximation of x^(1/2)
float SQRT(float x, int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1);
	assert (x > 0.0);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = sqrtf(x);
	else if (m == 1)
		//Bit shift approximation
		y = x;	//Change this!!!

	return y;
}

//Approximation of |x|
float ABS(float x, int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1);
	
	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = fabsf(x);
	else if (m == 1)
		//Bitwise operation
		y = x;	//Change this!!!

	return y;
}

//Approximation of ln(x)
float LOG(float x, int m)
{
	assert (m == 0 || m == 1 || m == 2);
	assert (x > 0.0);

	float y = 0.0;

	if (m == 0)
		//Defined int <math.h>
		y = logf(x);
	else if (m == 1)
		//Defined in "fastapprox.h"
		y = fastlog(x);
	else if (m == 2)
		//Defined in "fastapprox.h"
		y = fasterlog(x);

	return y;
}

//Approximation of sine(x)
float SIN(float x, int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1 || m == 2);
	assert (x > -M_PI && x < M_PI);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = sinf(x);
	else if (m == 1)
		//Defined in "fastapprox.h"
		y = fastsin(x);
	else if (m == 2)
		//Defined in "fastapprox.h"
		y = fastersin(x);

	return y;
}

//Approximation of cosine(x)
float COS(float x, int m)
{
	if (x == 0.0)
		return 1.0;

	assert (m == 0 || m == 1 || m == 2);
	assert (x > -M_PI && x < M_PI);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = cosf(x);
	else if (m == 1)
		//Defined in "fastapprox.h"
		y = fastcos(x);
	else if (m == 2)
		//Defined in "fastapprox.h"
		y = fastercos(x);

	return y;
}

//Approximation of tangent(x)
float TAN(float x, int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1 || m == 2);
	assert (x > -M_PI / 2.0 && x < M_PI / 2.0);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = tanf(x);
	else if (m == 1)
		//Defined in "fastapprox.h"
		y = fasttan(x);
	else if (m == 2)
		//Defined in "fastapprox.h"
		y = fastertan(x);

	return y;
}

//Approximation of arccosine(x)
float ACOS(float x, int m, enum Precision p)
{
	if (x == 0.0)
		return HALF_PI;

	assert (m == 0 || m == 1 || m == 2);
	assert (x > -1 && x < 1);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = acosf(x);
	else if (m == 1) {
		//Chebyshev Approximation
	} else if (m == 2)
		//Wolfram Series Representation (for |x| < 1)
		y = x;	//Change this!!!

	return y;
}

//Approximation of arctangent(x)
float ATAN(float x, int m, enum Precision p)
{
	if (x == 0.0)
		return x;
	else if (x == 1.0)	//Because the Wolfram series is not valid for x = 1.0
		return M_PI / 4.0;

	assert (m == 0 || m == 1 || m == 2);
	
	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = atanf(x);
	else if (m == 1)
		//Chebyshev Approximation
		y = x;	//Change this!!!
	else if (m == 2)
		//Wolfram Series Representation (for x != 1.0)
		y = x;	//Change this!!!

	return y;
}

//Approximation of sinh(x)
float SINH(float x, int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = sinhf(x);
	else if (m == 1)
		//Defined in "fastapprox.h"
		y = fastsinh(x);
	else if (m == 2)
		//Defined in "fastapprox.h"
		y = fastersinh(x);

	return y;
}

//Approximation of cosh(x)
float COSH(float x, int m)
{
	if (x == 0.0)
		return 1.0;

	assert (m == 0 || m == 1 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = coshf(x);
	else if (m == 1)
		//Defined in "fastapprox.h"
		y = fastcosh(x);
	else if (m == 2)
		//Defined in "fastapprox.h"
		y = fastercosh(x);

	return y;
}

//Approximation of arcsinh(x)
float ASINH(float x, int m, enum Precision p)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = asinhf(x);
	else if (m == 1)
		//Chebyshev Approximation
		y = x;	//Change this!!!
	else if (m == 2)
		//Wolfram Series Representation (for |x| < 1)
		y = x;	//Change this!!!

	return y;
}

//Approximation of arccosh(x)
float ACOSH(float x, int m, enum Precision p)
{
	if (x == 1.0)
		return 0.0;

	assert (m == 0 || m == 1 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = acoshf(x);
	else if (m == 1)
		//Chebyshev Approximation
		y = x;	//Change this!!!
	else if (m == 2)
		//Wolfram Series Approximation (for x > 1)
		y = x;	//Change this!!!

	return y;
}

//Approximation of the Gamma Function
float GAMMA(float x, int m)
{
	return x;
}

//Approximation of the Pochhammer symbol (x)_j
//The coefficient j must be a non-negative integer
float POCHHAMMER(float x, int j)
{
	return x;
}

//Approximates the Gauss Hypergeometric Function sol=2F1(a,b,c,z(x))
//The tolerance describes convergence of the series
//The precision 'p' describes how many terms are used in the series
bool _2F1(float (*z)(float x), float &sol, float x, float a, float b, float c, float tol, enum Precision p)
{
	return true;
}

#endif
