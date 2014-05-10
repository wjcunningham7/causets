#ifndef SUBROUTINES_H_
#define SUBROUTINES_H_

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

//Quicksort Algorithm
void quicksort(Node *nodes, int low, int high);
void swap(Node *n, Node *m);

//Newton-Raphson Algorithm
bool newton(double (*solve)(NewtonProperties *np), NewtonProperties *np, long *seed);

#endif
