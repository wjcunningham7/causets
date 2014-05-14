#ifndef FAST_MATH_H_
#define FAST_MATH_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

#define TOL (1e-28)

#define HALF_PI  1.57079632679489661923f
#define TWO_PI   6.28318530717958647692f

//Definitions of constants used in power series
#define ACOS_C0  HALF_PI
#define ACOS_C1 -1.06305396909634217923f
#define ACOS_C3  0.88385729242991187525f
#define ACOS_C5 -4.69522239734719040073f
#define ACOS_C7  7.39114112136511672686f
#define ACOS_C9 -4.02406572163211910684f

#define ACOS_W0  HALF_PI
#define ACOS_W1 -1.00000000000000000000f
#define ACOS_W3 -0.16666666666666666666f
#define ACOS_W5 -0.07500000000000000000f
#define ACOS_W7 -0.04464285714285714285f
#define ACOS_W9 -0.03038194444444444444f

#define ATAN_C1  1.04538834220118418960f
#define ATAN_C3 -0.39082098431982330905f
#define ATAN_C5  0.17944049001227966481f
#define ATAN_C7 -0.08419846479405229950f
#define ATAN_C9  0.02041955547722351862f

#define ATAN_V1  1.00000000000000000000f
#define ATAN_V3 -0.33333333333333333333f
#define ATAN_V5  0.20000000000000000000f
#define ATAN_V7 -0.14285714285714285714f
#define ATAN_V9  0.11111111111111111111f

#define ATAN_W0  HALF_PI
#define ATAN_W1 -1.00000000000000000000f
#define ATAN_W3  0.33333333333333333333f
#define ATAN_W5 -0.20000000000000000000f
#define ATAN_W7  0.14285714285714285714f
#define ATAN_W9 -0.11111111111111111111f

#define ASINH_W1   1.00000000000000000000f
#define ASINH_W3  -0.16666666666666666666f
#define ASINH_W5   0.07500000000000000000f
#define ASINH_W7  -0.04464285714285714285f
#define ASINH_W9   0.03038194444444444444f

#define ACOSH_W2  -0.25000000000000000000f
#define ACOSH_W4  -0.09375000000000000000f
#define ACOSH_W6  -0.10416666666666666666f
#define ACOSH_W8  -0.06835937500000000000f
#define ACOSH_W10 -0.04921875000000000000f

//Number of Terms in Series
enum Precision {
	LOW_PRECISION,		//O(x^4)
	HIGH_PRECISION,		//O(x^6)
	VERY_HIGH_PRECISION	//O(x^10)
};

//General Parameters:
// > Input x
// > Method m

//Power Functions
float POW2(const float x, const int m);
float POW3(const float x, const int m);
float POW(const float x, const float p, const int m);
float SQRT(const float x, const int m);

//Absolute Value, Natural Log, and Sign
float ABS(const float x, const int m);
float LOG(const float x, const int m);
float SGN(const float x, const int m);

//Trigonometric Functions
float SIN(const float x, const int m);
float COS(const float x, const int m);
float TAN(const float x, const int m);

//Inverse Trigonometric Functions
float ACOS(const float x, const int m, const enum Precision p);
float ATAN(const float x, const int m, const enum Precision p);

//Hyperbolic Functions
float SINH(const float x, const int m);
float COSH(const float x, const int m);

//Inverse Hyperbolic Functions
float ASINH(const float x, const int m, const enum Precision p);
float ACOSH(const float x, const int m, const enum Precision p);

//Statistical Functions
float GAMMA(const float x, const int m);
float POCHHAMMER(const float x, const int j);
bool _2F1(float (*z)(const float &x), float &sol, const float &x, const float a, const float b, const float c, const float tol, const enum Precision p);

#endif
