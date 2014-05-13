#ifndef FAST_MATH_H_
#define FAST_MATH_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>

#define TOL (1e-28)

#define HALF_PI  1.57079632679489661923
#define TWO_PI   6.28318530717958647692

//Definitions of constants used in functions
#define ACOS_C0  HALF_PI
#define ACOS_C1 -1.06305396909634217923
#define ACOS_C3  0.88385729242991187525

//Number of Terms in Series
enum Precision {
	LOW_PRECISION,		//4 Terms
	HIGH_PRECISION,		//6 Terms
	VERY_HIGH_PRECISION	//10 Terms
};

//General Parameters:
// > Input x
// > Method m

//Power Functions
float POW2(const float x, const int m);
float POW3(const float x, const int m);
float POW(const float x, const float p, const int m);
float SQRT(const float x, const int m);

//Absolute Value & Natural Log
float ABS(const float x, const int m);
float LOG(const float x, const int m);

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
