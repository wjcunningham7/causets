#ifndef CAUSET_OPERATIONS_HPP_
#define CAUSET_OPERATIONS_HPP_

#include <math.h>

//Variables used to evaluate Newton-Raphson Functions
struct NewtonProperties {
	NewtonProperties() : x(0.0), zeta(0.0), a(1.0), rval(0.0), tol(TOL), max(10000), N(10000), k(10), dim(4) {}
	NewtonProperties(double _zeta, double _tol, unsigned int _N, unsigned int _k, unsigned int _dim) : zeta(_zeta), tol(_tol), N(_N), k(_k), dim(_dim) {}
	NewtonProperties(double &_x, double _tol, int _max, unsigned int _N, unsigned int _k, unsigned int _dim) : x(_x), tol(_tol), max(_max), N(_N), k(_k), dim(_dim) {}
	NewtonProperties(double &_x, double _zeta, double _a, double _rval, double _tol, int _max, unsigned int _N, unsigned int _k, unsigned int _dim) : x(_x), zeta(_zeta), a(_a), rval(_rval), tol(_tol), max(_max), N(_N), k(_k), dim(_dim) {}

	double x;		//Current value to evaluate function at; solution stored here
	double zeta;		//Pi/2 - eta0
	double a;		//Pseudoscalar radius

	double rval;		//Value from RNG
	double tol;		//Tolerance

	int max;		//Max Iterations

	unsigned int N;	//Number of nodes
	unsigned int k;	//Expected average degrees
	unsigned int dim;	//Dimensions of spacetime
};

//Primary Newton-Raphson Functions
double solveZeta(NewtonProperties *np);
double solveTau(NewtonProperties *np);
double solvePhi(NewtonProperties *np);

//Secondary Newton-Raphson Functions
//double eta02D(double &x, unsigned int &N, unsigned int &k);
//double eta0Prime2D(double &x);

double zeta4D(double &x, unsigned int &N, unsigned int &k);
double zetaPrime4D(double &x);

double tau4D(double &x, double &zeta, double &a, double rval);
double tauPrime4D(double &x, double &zeta, double &a);

double phi4D(double &x, double rval);
double phiPrime4D(double &x);

//De Sitter Spatial Lengths
float X1(float &phi);
float X2(float &phi, float &chi);
float X3(float &phi, float &chi, float &theta);
float X4(float &phi, float &chi, float &theta);

//Temporal Transformations
float etaToTau(float eta, double a);
float tauToEta(float tau, double a);

#endif
