#ifndef OPERATIONS_CU_
#define OPERATIONS_CU_

#include "Causet.h"

//Variables used to evaluate Newton-Raphson Functions
struct NewtonProperties {
	NewtonProperties() : x(0.0), zeta(0.0), tau0(0.0), a(1.0), alpha(1.0), delta(1.0), rval(0.0), tol(TOL), max(10000), N_tar(10000), k_tar(10.0), dim(3) {}
	NewtonProperties(double _zeta, double _tol, int _N_tar, float _k_tar, int _dim) : zeta(_zeta), tol(_tol), N_tar(_N_tar), k_tar(_k_tar), dim(_dim) {}
	NewtonProperties(double &_x, double _tol, int _max, int _N_tar, float _k_tar, int _dim) : x(_x), tol(_tol), max(_max), N_tar(_N_tar), k_tar(_k_tar), dim(_dim) {}
	NewtonProperties(double &_x, double _tol, int _max, int _N_tar, double _alpha, double _delta) : x(_x), tol(_tol), max(_max), N_tar(_N_tar), alpha(_alpha), delta(_delta) {}
	NewtonProperties(double &_x, double _zeta, double _tau0, double _a, double _alpha, double _delta, double _rval, double _tol, int _max, int _N_tar, float _k_tar, int _dim) : x(_x), zeta(_zeta), tau0(_tau0), a(_a), alpha(_alpha), delta(_delta), rval(_rval), tol(_tol), max(_max), N_tar(_N_tar), k_tar(_k_tar), dim(_dim) {}

	double x;		//Current value to evaluate function at; solution stored here
	double zeta;		//Pi/2 - eta0
	double tau0;		//Rescaled age of universe
	double a;		//Pseudoscalar radius
	double alpha;		//Rescaled ratio of matter density to dark energy density
	double delta;		//Node density

	double rval;		//Value from RNG
	double tol;		//Tolerance

	int max;		//Max iterations

	int N_tar;		//Target number of nodes
	float k_tar;		//Target expected average degrees
	int dim;		//Dimensions of spacetime
};

//Primary Newton-Raphson Functions
double solveZeta(NewtonProperties *np);
double solveTau0(NewtonProperties *np);
double solveT(NewtonProperties *np);
double solveTau(NewtonProperties *np);
double solvePhi(NewtonProperties *np);

//Secondary Newton-Raphson Functions
double eta02D(double &x, int &N_tar, float &k_tar);
double eta0Prime2D(double &x);

double zeta4D(double &x, int &N_tar, float &k_tar);
double zetaPrime4D(double &x);

double tau0(double &x, int &N_tar, double &alpha, double &delta);
double tau0Prime(double &x);

double t4D(double &x, double &zeta, double &a, double rval);
double tPrime4D(double &x, double &zeta, double &a);

double tau(double &x, double &tau0, double rval);
double tauPrime(double &x, double &tau0);

double phi4D(double &x, double rval);
double phiPrime4D(double &x);

//De Sitter Spatial Lengths
float X1(float &phi);
float X2(float &phi, float &chi);
float X3(float &phi, float &chi, float &theta);
float X4(float &phi, float &chi, float &theta);

//Temporal Transformations
float etaToT(float eta, double a);
float tToEta(float t, double a);

//////////////////////////////////
//Primary Newton-Raphson Functions
//All O(1) Efficiency
//////////////////////////////////

//Returns zeta Residual
inline double solveZeta(NewtonProperties *np)
{
	return ((np->dim == 1) ?
		-1.0 * eta02D(np->x, np->N_tar, np->k_tar) / eta0Prime2D(np->x) : 
		-1.0 * zeta4D(np->x, np->N_tar, np->k_tar) / zetaPrime4D(np->x));
}

//Returns tau0 Residual
inline double solveTau0(NewtonProperties *np)
{
	return (-1.0 * tau0(np->x, np->N_tar, np->alpha, np->delta) / tau0Prime(np->x));
}

//Returns t Residual
inline double solveT(NewtonProperties *np)
{
	return (-1.0 * t4D(np->x, np->zeta, np->a, np->rval) / tPrime4D(np->x, np->zeta, np->a));
}

//Returns tau Residual
inline double solveTau(NewtonProperties *np)
{
	return (-1.0 * tau(np->x, np->tau0, np->rval) / tauPrime(np->x, np->tau0));
}

//Returns phi Residual
inline double solvePhi(NewtonProperties *np)
{
	return (-1.0 * phi4D(np->x, np->rval) / phiPrime4D(np->x));
}

////////////////////////////////////
//Secondary Newton-Raphson Functions
////////////////////////////////////

inline double eta02D(double &x, int &N_tar, float &k_tar)
{
	return (((2.0 / M_PI) * (((x / tan(x)) + log(1.0 / cos(x)) - 1.0) / tan(x))) - (k_tar / N_tar));
}

inline double eta0Prime2D(double &x)
{
	return ((2.0 / M_PI) * (((1.0 / tan(x)) * ((1.0 / tan(x)) - (x / (sin(x) * sin(x))) + tan(x))) - ((1.0 / (sin(x) * sin(x))) * (log(1.0 / cos(x)) + (x / tan(x)) - 1.0))));
}

inline double zeta4D(double &x, int &N_tar, float &k_tar)
{
	return (((2.0 / (3.0 * M_PI)) * (12.0 * (((M_PI / 2.0) - x) * tan(x) + log(1.0 / sin(x))) + ((6.0 * log(1.0 / sin(x)) - 5.0) / (sin(x) * sin(x))) - 7) / ((2.0 + (1.0 / (sin(x) * sin(x)))) * (2.0 + (1.0 / (sin(x) * sin(x)))) / tan(x))) - (k_tar / N_tar));
}

inline double zetaPrime4D(double &x)
{
	return ((3.0 * (cos(5.0 * x) - 32.0 * (M_PI - 2.0 * x) * sin(x) * sin(x) * sin(x)) + cos(x) * (84.0 - 72.0 * log(1.0 / sin(x))) + cos(3.0 * x) * (24.0 * log(1.0 / sin(x)) - 31.0)) / (-4.0 * M_PI * sin(x) * sin(x) * sin(x) * sin(x) * cos(x) * cos(x) * cos(x) * pow(2.0 + (1.0 / (sin(x) * sin(x))), 3.0)));
}

inline double tau0(double &x, int &N_tar, double &alpha, double &delta)
{
	return (sinh(3.0 * x) - 3.0 * x - 3.0 * N_tar / (M_PI * M_PI * delta * alpha * alpha * alpha));
}

inline double tau0Prime(double &x)
{
	return (3.0 * (cosh(3.0 * x) - 1.0));
}

inline double t4D(double &x, double &zeta, double &a, double rval)
{
	return ((((2.0 + cosh(x / a) * cosh(x / a)) * sinh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta))) - rval);
}

inline double tPrime4D(double &x, double &zeta, double &a)
{
	return ((3.0 * cosh(x / a) * cosh(x / a) * cosh(x / a) * cosh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta)));
}

inline double tau(double &x, double &tau0, double rval)
{
	return ((sinh(3.0 * x) - 3.0 * x) / (sinh(3.0 * tau0) - 3.0 * tau0) - rval);
}

inline double tauPrime(double &x, double &tau0)
{
	return (6.0 * sinh(1.5 * x) / (sinh(3.0 * tau0) - 3.0 * tau0));
}

inline double phi4D(double &x, double rval)
{
	return (((x - sin(x) * cos(x)) / M_PI) - rval);
}

inline double phiPrime4D(double &x)
{
	return ((2 / M_PI) * sin(x) * sin(x));
}

///////////////////////////
//De Sitter Spatial Lengths
///////////////////////////

//X1 Coordinate of de Sitter Metric
inline float X1(float &phi)
{
	return cosf(phi);
}

//X2 Coordinate of de Sitter Metric
inline float X2(float &phi, float &chi)
{
	return (sinf(phi) * cosf(chi));
}

//X3 Coordinate of de Sitter Metric
inline float X3(float &phi, float &chi, float &theta)
{
	return (sinf(phi) * sinf(chi) * cosf(theta));
}

//X4 Coordinate of de Sitter Metric
inline float X4(float &phi, float &chi, float &theta)
{
	return (sinf(phi) * sinf(chi) * sinf(theta));
}

//////////////////////////
//Temporal Transformations
//////////////////////////

//Conformal to Rescaled Time
inline float etaToT(float eta, double a)
{
	return (acoshf(1.0 / cosf(eta)) * a);
}

//Rescaled to Conformal Time
inline float tToEta(float t, double a)
{
	return (acosf(1.0 / coshf(t / a)));
}

#endif
