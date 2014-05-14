#include <boost/math/special_functions/gamma.hpp>

#include "fastapprox.h"
#include "FastMath.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Approximation of x^2
float POW2(const float x, const int m)
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
float POW3(const float x, const int m)
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
float POW(const float x, const float p, const int m)
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
float SQRT(const float x, const int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1);
	assert (x > 0.0);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = sqrtf(x);
	else if (m == 1) {
		//Bit shift approximation
		unsigned int i = *(unsigned int*) &x;
		i += 127 << 23;
		i >>= 1;
		y = *(float*) &i;
	}

	return y;
}

//Approximation of |x|
float ABS(const float x, const int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1);
	
	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = fabsf(x);
	else if (m == 1) {
		//Bitwise operation
		int i = *(int*) &x;
		i &= 0x7FFFFFFF;
		y = *(float*) &i;
	}

	return y;
}

//Approximation of ln(x)
float LOG(const float x, const int m)
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

//Returns sign(x)
float SGN(const float x, const int m)
{	
	//Cannot be zero
	assert (ABS(round(x), 0) > TOL);
	assert (m == 0 || m == 1);

	float y = 1.0f;

	if (m == 0)
		if (y < 0.0)
			y = -1.0f;
	else if (m == 1)
		(int&)y |= ((int&)y & 0x80000000);

	return y;
}

//Approximation of sine(x)
float SIN(const float x, const int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = sinf(x);
	else if (m == 1) {
		//Defined in "fastapprox.h"
		assert (x > -M_PI && x < M_PI);
		y = fastsin(x);
	} else if (m == 2) {
		//Defined in "fastapprox.h"
		assert (x > -M_PI && x < M_PI);
		y = fastersin(x);
	}

	return y;
}

//Approximation of cosine(x)
float COS(const float x, const int m)
{
	if (x == 0.0)
		return 1.0;

	assert (m == 0 || m == 1 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = cosf(x);
	else if (m == 1) {
		//Defined in "fastapprox.h"
		assert (x > -M_PI && x < M_PI);
		y = fastcos(x);
	} else if (m == 2) {
		//Defined in "fastapprox.h"
		assert (x > -M_PI && x < M_PI);
		y = fastercos(x);
	}

	return y;
}

//Approximation of tangent(x)
float TAN(const float x, const int m)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 1 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = tanf(x);
	else if (m == 1) {
		//Defined in "fastapprox.h"
		y = fasttan(x);
		assert (x > -M_PI / 2.0 && x < M_PI / 2.0);
	} else if (m == 2) {
		//Defined in "fastapprox.h"
		assert (x > -M_PI / 2.0 && x < M_PI / 2.0);
		y = fastertan(x);
	}

	return y;
}

//Approximation of arccosine(x)
float ACOS(const float x, const int m, const enum Precision p)
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
		float _x2 = POW2(x, 0);
		if (p == LOW_PRECISION)
			y = ACOS_C0 + x * (ACOS_C3 * _x2 + ACOS_C1);
		else if (p == HIGH_PRECISION)
			y = ACOS_C0 + x * (_x2 * (ACOS_C5 * _x2 + ACOS_C3) + ACOS_C1);
		else if (p == VERY_HIGH_PRECISION)
			y = ACOS_C0 + x * (_x2 * (_x2 * (_x2 * (ACOS_C9 * _x2 + ACOS_C7) + ACOS_C5) + ACOS_C3) + ACOS_C1);
	} else if (m == 2) {
		//Wolfram Series Representation (for |x| < 1)
		float _x2 = POW2(x, 0);
		if (p == LOW_PRECISION)
			y = ACOS_W0 + x * (ACOS_W3 * _x2 + ACOS_C1);
		else if (p == HIGH_PRECISION)
			y = ACOS_W0 + x * (_x2 * (ACOS_W5 * _x2 + ACOS_W3) + ACOS_W1);
		else if (p == VERY_HIGH_PRECISION)
			y = ACOS_W0 + x * (_x2 * (_x2 * (_x2 * (ACOS_W9 * _x2 + ACOS_W7) + ACOS_W5) + ACOS_W3) + ACOS_W1);
	}

	return y;
}

//Approximation of arctangent(x)
float ATAN(const float x, const int m, const enum Precision p)
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
	else if (m == 1) {
		//Chebyshev Approximation
		float _x2 = POW2(x, 0);
		if (p == LOW_PRECISION)
			y = x * (ATAN_C3 * _x2 + ATAN_C1);
		else if (p == HIGH_PRECISION)
			y = x * (_x2 * (ATAN_C5 * _x2 + ATAN_C3) + ATAN_C1);
		else if (p == VERY_HIGH_PRECISION)
			y = x * (_x2 * (_x2 * (_x2 * (ATAN_C9 * _x2 + ATAN_C7) + ATAN_C5) + ATAN_C3) + ATAN_C1);
	} else if (m == 2) {
		//Wolfram Series Representation (for x != 1.0)
		if (SGN(x, 0) == -1.0f) {
			float _x2 = POW2(x, 0);
			if (p == LOW_PRECISION)
				y = x * (ATAN_V3 * _x2 + ATAN_V1);
			else if (p == HIGH_PRECISION)
				y = x * (_x2 * (ATAN_V5 * _x2 + ATAN_V3) + ATAN_V1);
			else if (p == VERY_HIGH_PRECISION)
				y = x * (_x2 * (_x2 * (_x2 * (ATAN_V9 * _x2 + ATAN_V7) + ATAN_V5) + ATAN_V3) + ATAN_V1);
		} else {
			float _x2minus = 1.0 / POW2(x, 0);
			if (p == LOW_PRECISION)
				y = ATAN_W0 * SGN(x, 0) + (ATAN_W3 * _x2minus + ATAN_W1) / x;
			else if (p == HIGH_PRECISION)
				y = ATAN_W0 * SGN(x, 0) + (_x2minus * (ATAN_W5 * _x2minus + ATAN_W3) + ATAN_W1) / x;
			else if (p == VERY_HIGH_PRECISION)
				y = ATAN_W0 * SGN(x, 0) + (_x2minus * (_x2minus * (_x2minus * (ATAN_W9 * _x2minus + ATAN_W7) + ATAN_W5) + ATAN_W3) + ATAN_W1) / x;
		}
	}

	return y;
}

//Approximation of sinh(x)
float SINH(const float x, const int m)
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
float COSH(const float x, const int m)
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
//Note the Chebyshev approximation is not fast here
float ASINH(const float x, const int m, const enum Precision p)
{
	if (x == 0.0)
		return x;

	assert (m == 0 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = asinhf(x);
	else if (m == 1)
		//Chebyshev Approximation (purposely not implemented)
		y = x;
	else if (m == 2) {
		//Wolfram Series Representation (for |x| < 1)
		assert (ABS(x, 0) < 1.0);
		float _x2 = POW2(x, 0);
		if (p == LOW_PRECISION)
			y = x * (ASINH_W3 * _x2 + ASINH_W1);
		else if (p == HIGH_PRECISION)
			y = x * (_x2 * (ASINH_W5 * _x2 + ASINH_W3) + ASINH_W1);
		else if (p == VERY_HIGH_PRECISION)
			y = x * (_x2 * (_x2 * (_x2 * (ASINH_W9 * _x2 + ASINH_W7) + ASINH_W5) + ASINH_W3) + ASINH_W1);
	}

	return y;
}

//Approximation of arccosh(x)
//Note the Chebyshev approximation is not fast here
float ACOSH(const float x, const int m, const enum Precision p)
{
	if (x == 1.0)
		return 0.0;

	assert (m == 0 || m == 1 || m == 2);

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = acoshf(x);
	else if (m == 1)
		//Chebyshev Approximation (purposely not implemented)
		y = x;
	else if (m == 2) {
		//Wolfram Series Approximation (for x > 1)
		assert (ABS(x, 0) > 1.0);
		float _x2minus = 1.0 / POW2(x, 0);
		y += LOG(2.0 * x, 0);
		if (p == LOW_PRECISION)
			y += _x2minus * (ACOSH_W4 * _x2minus + ACOSH_W2);
		else if (p == HIGH_PRECISION)
			y += _x2minus * (_x2minus * (ACOSH_W6 * _x2minus + ACOSH_W4) + ACOSH_W2);
		else if (p == VERY_HIGH_PRECISION)
			y += _x2minus * (_x2minus * (_x2minus * (_x2minus * (ACOSH_W10 * _x2minus + ACOSH_W8) + ACOSH_W6) + ACOSH_W4) + ACOSH_W2);
	}

	return y;
}

//Approximation of the Gamma Function
float GAMMA(const float x, const int m)
{
	if (x == 1.0 || x == 2.0)
		return 1.0;

	assert (m == 0 || m == 1);
	//Gamma(0) is undefined
	assert (x != 0.0);
	assert (!(x > 0.0 && floor(x) < TOL));
	assert (!(x < 0.0 && ABS(ceil(x), 0) < TOL));
	//Gamma not defined for negative integers
	assert (!(x < 0.0 && ABS(x - round(x), 0) < TOL));

	float y = 0.0;

	if (m == 0)
		//Defined in <math.h>
		y = tgamma(x);
	else if (m == 1)
		//Lanczos Approximation
		//Defined in Boost library
		y = boost::math::tgamma1pm1(x);

	return y;
}

//Approximation of the Pochhammer symbol (x)_j
//The coefficient j must be a non-negative integer
float POCHHAMMER(const float x, const int j)
{
	assert (j >= 0);

	float y = 0.0;
	int m = 0;

	y = GAMMA(x + j, m) / GAMMA(x, m);

	return y;
}

//Approximates the Gauss Hypergeometric Function sol=2F1(a,b,c,z(x))
//The tolerance describes convergence of the series
//The precision 'p' describes how many terms are used in the series
bool _2F1(float (*z)(const float &x), float &sol, const float &x, const float a, const float b, const float c, const float tol, const enum Precision p)
{
	//No null pointers
	assert (z != NULL);

	//The value c - a - b must be greater than zero and not an integer
	if (c - a - b <= 0 || ABS(round(c - a - b) - (c - a - b), 0) < TOL)
		return false;

	//Valid region for power series approximation
	if (z(x) <= 1.0) {
		int nterms;
		int i;

		if (p == LOW_PRECISION)
			nterms = 3;
		else if (p == HIGH_PRECISION)
			nterms = 5;
		else if (p == VERY_HIGH_PRECISION)
			nterms = 7;

		for (i = 0; i < nterms; i++)
			sol += POCHHAMMER(a, i) * POCHHAMMER(b, i) * POW(z(x), i, 0) / (POCHHAMMER(c, i) * GAMMA(i + 1, 0));
	} else if (z(x) > 1.0) {
		//Use transformation described in notes
	}

	return true;
}
