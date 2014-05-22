#include "Operations.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Returns zeta Residual
double solveZeta(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	//No null pointers
	assert (p4 != NULL);	//k_tar
	assert (p5 != NULL);	//N_tar
	assert (p6 != NULL);	//dim

	//Variables in the correct ranges
	assert (*p4 > 0.0);
	assert (*p5 > 0);
	assert (*p6 == 1 || *p6 == 3);

	return ((*p6 == 1) ?
		-1.0 * eta02D(x, *p5, *p4) / eta0Prime2D(x) : 
		-1.0 * zeta4D(x, *p5, *p4) / zetaPrime4D(x));
}

//Returns tau0 Residual
double solveTau0(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	//No null pointers
	assert (p1 != NULL);	//alpha
	assert (p2 != NULL);	//delta
	assert (p5 != NULL);	//N_tar

	//Variables in the correct ranges
	assert (*p1 > 0.0);
	assert (*p2 > 0);
	assert (*p5 > 0);

	return (-1.0 * tau0(x, *p5, *p1, *p2) / tau0Prime(x));
}

//Returns t Residual
double solveT(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	//No null pointers
	assert (p1 != NULL);	//zeta
	assert (p2 != NULL);	//a
	assert (p3 != NULL);	//rval

	//Variables in the correct ranges
	assert (*p1 > 0.0 && *p1 < static_cast<double>(HALF_PI));
	assert (*p2 > 0.0);
	assert (*p3 > 0.0 && *p3 < 1.0);

	return (-1.0 * t4D(x, *p1, *p2, *p3) / tPrime4D(x, *p1, *p2));
}

//Returns tau Residual
double solveTau(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	//No null pointers
	assert (p1 != NULL);	//tau0
	assert (p2 != NULL);	//rval

	//Variables in the correct ranges
	assert (*p1 > 0.0);
	assert (*p2 > 0.0 && *p2 < 1.0);

	return (-1.0 * tau(x, *p1, *p2) / tauPrime(x, *p1));
}

//Returns phi Residual
double solvePhi(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	//No null pointers
	assert (p1 != NULL);	//rval

	//Variables in correct ranges
	assert (*p1 > 0.0 && *p1 < 1.0);

	return (-1.0 * phi4D(x, *p1) / phiPrime4D(x));
}

double eta02D(const double &x, const int &N_tar, const float &k_tar)
{
	double _tanx = static_cast<double>(TAN(static_cast<float>(x), FAST));
	return ((x / _tanx - static_cast<double>(LOG(COS(static_cast<float>(x), FAST), FAST)) - 1.0) / (_tanx * static_cast<double>(HALF_PI))) - (static_cast<double>(k_tar) / static_cast<double>(N_tar));
}

double eta0Prime2D(const double &x)
{
	double _cotx = 1.0 / static_cast<double>(TAN(static_cast<float>(x), FAST));
	double _cscx2 = 1.0 / static_cast<double>(POW2(SIN(static_cast<float>(x), FAST), EXACT));
	double _lnsecx = -1.0 * static_cast<double>(LOG(COS(static_cast<float>(x), FAST), FAST));

	return (_cotx * (_cotx - x * _cscx2) + 1 - _cscx2 * (_lnsecx + x * _cotx - 1)) / static_cast<double>(HALF_PI);
}

double zeta4D(const double &x, const int &N_tar, const float &k_tar)
{
	double _tanx = static_cast<double>(TAN(static_cast<float>(x), FAST));
	double _cscx2 = 1.0 / static_cast<double>(POW2(SIN(static_cast<float>(x), FAST), EXACT));
	double _lncscx = -1.0 * static_cast<double>(LOG(SIN(static_cast<float>(x), FAST), FAST));

	return _tanx * (12.0 * (x * _tanx + _lncscx) + (6.0 * _lncscx - 5.0) * _cscx2 - 7.0) / (3.0 * static_cast<double>(HALF_PI) * static_cast<double>(POW2(2.0 + static_cast<float>(_cscx2), EXACT))) - static_cast<double>(k_tar) / static_cast<double>(N_tar);
}

double zetaPrime4D(const double &x)
{
	double _cscx2 = 1.0 / static_cast<double>(POW2(SIN(static_cast<float>(x), FAST), EXACT));
	double _sinx3 = static_cast<double>(POW3(SIN(static_cast<float>(x), FAST), EXACT));
	double _sinx4 = static_cast<double>(POW(SIN(static_cast<float>(x), FAST), 4.0, FAST));
	double _cosx3 = static_cast<double>(POW3(COS(static_cast<float>(x), FAST), EXACT));
	double _lncscx = -1.0 * static_cast<double>(LOG(SIN(static_cast<float>(x), FAST), FAST));

	return (3.0 * (static_cast<double>(COS(5.0 * static_cast<float>(x), FAST)) - 32.0 * (static_cast<double>(M_PI) - 2.0 * x) * _sinx3) + static_cast<double>(COS(static_cast<float>(x), FAST)) * (84.0 - 72.0 * _lncscx) + static_cast<double>(COS(3.0 * static_cast<float>(x), FAST)) * (24.0 * _lncscx - 31.0)) / (-4.0 * static_cast<double>(M_PI) * _sinx4 * _cosx3 * static_cast<double>(POW3((2.0 + static_cast<float>(_cscx2)), EXACT)));
}

double tau0(const double &x, const int &N_tar, const double &alpha, const double &delta)
{
	return static_cast<double>(SINH(3.0 * static_cast<float>(x), FAST)) - 3.0 * (x + static_cast<double>(N_tar) / (static_cast<double>(POW2(static_cast<float>(M_PI), EXACT)) * delta * static_cast<double>(POW3(static_cast<float>(alpha), EXACT))));
}

double tau0Prime(const double &x)
{
	return 3.0 * (static_cast<double>(COSH(3.0 * static_cast<float>(x), FAST)) - 1.0);
}

double t4D(const double &x, const double &zeta, const double &a, const double &rval)
{
	double _coshx2 = static_cast<double>(POW(COSH(static_cast<float>(x / a), FAST), 4.0, FAST));
	double _sinz2 = static_cast<double>(POW2(SIN(static_cast<float>(zeta), FAST), EXACT));

	return (2.0 + _coshx2) * static_cast<double>(SINH(static_cast<float>(x / a), FAST)) * static_cast<double>(TAN(static_cast<float>(zeta), FAST)) * _sinz2 / (2.0 * _sinz2 + 1) - rval;

	//return ((((2.0 + cosh(x / a) * cosh(x / a)) * sinh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta))) - rval);
}

double tPrime4D(const double &x, const double &zeta, const double &a)
{
	double _coshx4 = static_cast<double>(POW(COSH(static_cast<float>(x / a), FAST), 4.0, FAST));
	double _sinz2 = static_cast<double>(POW2(SIN(static_cast<float>(zeta), FAST), EXACT));

	return 3.0 * _coshx4 * static_cast<double>(TAN(static_cast<float>(zeta), FAST)) * _sinz2 / (2.0 * _sinz2 + 1);

	//return ((3.0 * cosh(x / a) * cosh(x / a) * cosh(x / a) * cosh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta)));
}

double tau(const double &x, const double &tau0, const double &rval)
{
	return (static_cast<double>(SINH(3.0 * static_cast<float>(x), FAST)) - 3.0 * x) / (static_cast<double>(SINH(3.0 * static_cast<float>(tau0), FAST)) - 3.0 * tau0) - rval;

	//return ((sinh(3.0 * x) - 3.0 * x) / (sinh(3.0 * tau0) - 3.0 * tau0) - rval);
}

double tauPrime(const double &x, const double &tau0)
{
	return 6.0 * static_cast<double>(POW2(SINH(1.5 * static_cast<float>(x), FAST), EXACT)) / (static_cast<double>(SINH(3.0 * static_cast<float>(tau0), FAST)) - 3.0 * tau0);

	//return (6.0 * sinh(1.5 * x) / (sinh(3.0 * tau0) - 3.0 * tau0));
}

double phi4D(const double &x, const double &rval)
{
	return (2.0 * x	- static_cast<double>(SIN(2.0 * static_cast<float>(x), FAST))) / static_cast<double>(TWO_PI) - rval;

	//return (((x - sin(x) * cos(x)) / M_PI) - rval);
}

double phiPrime4D(const double &x)
{
	return static_cast<double>(POW2(SIN(static_cast<float>(x), FAST), EXACT)) / static_cast<double>(HALF_PI);

	//return ((2 / M_PI) * sin(x) * sin(x));
}

//Math Function for Gauss Hypergeometric Function
//This is used to solve for a more exact solution than the one provided
//by numerical integration using the tauToEtaUniverse function
double _2F1_z(const double &tau)
{
	return static_cast<double>(POW2(COSH(1.5 * static_cast<float>(tau), FAST), EXACT));
}

//De Sitter Spatial Lengths

//X1 Coordinate of de Sitter Metric
float X1(const float &phi)
{
	return COS(phi, FAST);
}

//X2 Coordinate of de Sitter Metric
float X2(const float &phi, const float &chi)
{
	return SIN(phi, FAST) * COS(chi, FAST);
}

//X3 Coordinate of de Sitter Metric
float X3(const float &phi, const float &chi, const float &theta)
{
	return SIN(phi, FAST) * SIN(chi, FAST) * COS(theta, FAST);
}

//X4 Coordinate of de Sitter Metric
float X4(const float &phi, const float &chi, const float &theta)
{
	return SIN(phi, FAST) * SIN(chi, FAST) * SIN(theta, FAST);
}

//Temporal Transformations

//Conformal to Minkowski Time
float etaToT(const float eta, const double a)
{
	return ACOSH(1.0 / COS(eta, FAST), INTEGRATION, VERY_HIGH_PRECISION) * static_cast<float>(a);
	
	//return acoshf(1.0 / cosf(eta)) * a;
}

//Minkowski to Conformal Time
float tToEta(const float t, const double a)
{
	return ACOS(1.0 / COSH(t / static_cast<float>(a), FAST), INTEGRATION, VERY_HIGH_PRECISION);

	//return acosf(1.0 / coshf(t / a));
}

//Minkowski to Conformal Time (Universe)
//For use with GNU Scientific Library
double tToEtaUniverse(double t, void *params)
{
	return static_cast<double>(POW(SINH(1.5 * static_cast<float>(t), FAST), (-2.0 / 3.0), FAST));
}
