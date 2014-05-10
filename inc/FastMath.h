#ifndef FAST_MATH_H_
#define FAST_MATH_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>

#define HALF_PI 1.57079632679489661923
#define TWO_PI  6.28318530717958647692

//These are approximations of simple math subroutines

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
float POW2(float x, int m);
float POW3(float x, int m);
float POW(float x, float p, int m);
float SQRT(float x, int m);

//Absolute Value & Natural Log
float ABS(float x, int m);
float LOG(float x, int m);

//Trigonometric Functions
float SIN(float x, int m);
float COS(float x, int m);
float TAN(float x, int m);

//Inverse Trigonometric Functions
float ACOS(float x, int m, enum Precision p);
float ATAN(float x, int m, enum Precision p);

//Hyperbolic Functions
float SINH(float x, int m);
float COSH(float x, int m);

//Inverse Hyperbolic Functions
float ASINH(float x, int m, enum Precision p);
float ACOSH(float x, int m, enum Precision p);

//Statistical Functions
float GAMMA(float x, int m);
float POCHHAMMER(float x, int j);
bool _2F1(float (*z)(float x), float &sol, float x, float a, float b, float c, float tol, enum Precision p);

#endif
