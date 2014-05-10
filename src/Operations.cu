#ifndef OPERATIONS_CU_
#define OPERATIONS_CU_

#include "Operations.h"

//////////////////////////////////
//Primary Newton-Raphson Functions
//All O(1) Efficiency
//////////////////////////////////

//Returns zeta Residual
double solveZeta(NewtonProperties *np)
{
	return ((np->dim == 1) ?
		-1.0 * eta02D(np->x, np->N_tar, np->k_tar) / eta0Prime2D(np->x) : 
		-1.0 * zeta4D(np->x, np->N_tar, np->k_tar) / zetaPrime4D(np->x));
}

//Returns tau0 Residual
double solveTau0(NewtonProperties *np)
{
	return (-1.0 * tau0(np->x, np->N_tar, np->alpha, np->delta) / tau0Prime(np->x));
}

//Returns t Residual
double solveT(NewtonProperties *np)
{
	return (-1.0 * t4D(np->x, np->zeta, np->a, np->rval) / tPrime4D(np->x, np->zeta, np->a));
}

//Returns tau Residual
double solveTau(NewtonProperties *np)
{
	return (-1.0 * tau(np->x, np->tau0, np->rval) / tauPrime(np->x, np->tau0));
}

//Returns phi Residual
double solvePhi(NewtonProperties *np)
{
	return (-1.0 * phi4D(np->x, np->rval) / phiPrime4D(np->x));
}

////////////////////////////////////
//Secondary Newton-Raphson Functions
////////////////////////////////////

double eta02D(double &x, int &N_tar, float &k_tar)
{
	return (((2.0 / M_PI) * (((x / tan(x)) + log(1.0 / cos(x)) - 1.0) / tan(x))) - (k_tar / N_tar));
}

double eta0Prime2D(double &x)
{
	return ((2.0 / M_PI) * (((1.0 / tan(x)) * ((1.0 / tan(x)) - (x / (sin(x) * sin(x))) + tan(x))) - ((1.0 / (sin(x) * sin(x))) * (log(1.0 / cos(x)) + (x / tan(x)) - 1.0))));
}

double zeta4D(double &x, int &N_tar, float &k_tar)
{
	return (((2.0 / (3.0 * M_PI)) * (12.0 * ((HALF_PI - x) * tan(x) + log(1.0 / sin(x))) + ((6.0 * log(1.0 / sin(x)) - 5.0) / (sin(x) * sin(x))) - 7) / ((2.0 + (1.0 / (sin(x) * sin(x)))) * (2.0 + (1.0 / (sin(x) * sin(x)))) / tan(x))) - (k_tar / N_tar));
}

double zetaPrime4D(double &x)
{
	return ((3.0 * (cos(5.0 * x) - 32.0 * (M_PI - 2.0 * x) * sin(x) * sin(x) * sin(x)) + cos(x) * (84.0 - 72.0 * log(1.0 / sin(x))) + cos(3.0 * x) * (24.0 * log(1.0 / sin(x)) - 31.0)) / (-4.0 * M_PI * sin(x) * sin(x) * sin(x) * sin(x) * cos(x) * cos(x) * cos(x) * pow(2.0 + (1.0 / (sin(x) * sin(x))), 3.0)));
}

double tau0(double &x, int &N_tar, double &alpha, double &delta)
{
	return (sinh(3.0 * x) - 3.0 * x - 3.0 * N_tar / (M_PI * M_PI * delta * alpha * alpha * alpha));
}

double tau0Prime(double &x)
{
	return (3.0 * (cosh(3.0 * x) - 1.0));
}

double t4D(double &x, double &zeta, double &a, double rval)
{
	return ((((2.0 + cosh(x / a) * cosh(x / a)) * sinh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta))) - rval);
}

double tPrime4D(double &x, double &zeta, double &a)
{
	return ((3.0 * cosh(x / a) * cosh(x / a) * cosh(x / a) * cosh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta)));
}

double tau(double &x, double &tau0, double rval)
{
	return ((sinh(3.0 * x) - 3.0 * x) / (sinh(3.0 * tau0) - 3.0 * tau0) - rval);
}

double tauPrime(double &x, double &tau0)
{
	return (6.0 * sinh(1.5 * x) / (sinh(3.0 * tau0) - 3.0 * tau0));
}

double phi4D(double &x, double rval)
{
	return (((x - sin(x) * cos(x)) / M_PI) - rval);
}

double phiPrime4D(double &x)
{
	return ((2 / M_PI) * sin(x) * sin(x));
}

///////////////////////////
//De Sitter Spatial Lengths
///////////////////////////

//X1 Coordinate of de Sitter Metric
float X1(float &phi)
{
	return cosf(phi);
}

//X2 Coordinate of de Sitter Metric
float X2(float &phi, float &chi)
{
	return (sinf(phi) * cosf(chi));
}

//X3 Coordinate of de Sitter Metric
float X3(float &phi, float &chi, float &theta)
{
	return (sinf(phi) * sinf(chi) * cosf(theta));
}

//X4 Coordinate of de Sitter Metric
float X4(float &phi, float &chi, float &theta)
{
	return (sinf(phi) * sinf(chi) * sinf(theta));
}

//////////////////////////
//Temporal Transformations
//////////////////////////

//Conformal to Rescaled Time
float etaToT(float eta, double a)
{
	return (acoshf(1.0 / cosf(eta)) * a);
}

//Rescaled to Conformal Time
float tToEta(float t, double a)
{
	return (acosf(1.0 / coshf(t / a)));
}

#endif
