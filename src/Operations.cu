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
	assert (*p1 > 0.0 && *p1 < HALF_PI);
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
	float _tanx = TAN(x, 0);
	return ((x / _tanx - LOG(COS(x, 0), 0) - 1.0) / (_tanx * HALF_PI)) - (k_tar / N_tar);
}

double eta0Prime2D(const double &x)
{
	float _cotx = 1.0 / TAN(x, 0);
	float _cscx2 = 1.0 / POW2(SIN(x, 0), 0);
	float _lnsecx = -1.0 * LOG(COS(x, 0), 0);

	return (_cotx * (_cotx - x * _cscx2) + 1 - _cscx2 * (_lnsecx + x * _cotx - 1)) / HALF_PI;
}

double zeta4D(const double &x, const int &N_tar, const float &k_tar)
{
	float _tanx = TAN(x, 0);
	float _cscx2 = 1.0 / POW2(SIN(x, 0), 0);
	float _lncscx = -1.0 * LOG(SIN(x, 0), 0);

	return _tanx * (12.0 * (x * _tanx + _lncscx) + (6.0 * _lncscx - 5.0) * _cscx2 - 7.0) / (3.0 * HALF_PI * POW2(2.0 + _cscx2, 0)) - k_tar / N_tar;
}

double zetaPrime4D(const double &x)
{
	float _cscx2 = 1.0 / POW2(SIN(x, 0), 0);
	float _sinx3 = POW3(SIN(x, 0), 0);
	float _sinx4 = POW(SIN(x, 0), 4, 0);
	float _cosx3 = POW3(COS(x, 0), 0);
	float _lncscx = -1.0 * LOG(SIN(x, 0), 0);

	return (3.0 * (COS(5.0 * x, 0) - 32.0 * (M_PI - 2.0 * x) * _sinx3) + COS(x, 0) * (84.0 - 72.0 * _lncscx) + COS(3.0 * x, 0) * (24.0 * _lncscx - 31.0)) / (-4.0 * M_PI * _sinx4 * _cosx3 * POW3((2.0 + _cscx2), 0));
}

double tau0(const double &x, const int &N_tar, const double &alpha, const double &delta)
{
	return SINH(3.0 * x, 0) - 3.0 * (x + N_tar / (POW2(M_PI, 0) * delta * POW3(alpha, 0)));
}

double tau0Prime(const double &x)
{
	return 3.0 * (COSH(3.0 * x, 0) - 1.0);
}

double t4D(const double &x, const double &zeta, const double &a, const double &rval)
{
	float _coshx2 = POW(COSH(x / a, 0), 4.0, 0);
	float _sinz2 = POW2(SIN(zeta, 0), 0);

	return (2.0 + _coshx2) * SINH(x / a, 0) * TAN(zeta, 0) * _sinz2 / (2.0 * _sinz2 + 1) - rval;

	//return ((((2.0 + cosh(x / a) * cosh(x / a)) * sinh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta))) - rval);
}

double tPrime4D(const double &x, const double &zeta, const double &a)
{
	float _coshx4 = POW(COSH(x / a, 0), 4.0, 0);
	float _sinz2 = POW2(SIN(zeta, 0), 0);

	return 3.0 * _coshx4 * TAN(zeta, 0) * _sinz2 / (2.0 * _sinz2 + 1);

	//return ((3.0 * cosh(x / a) * cosh(x / a) * cosh(x / a) * cosh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta)));
}

double tau(const double &x, const double &tau0, const double &rval)
{
	return (SINH(3.0 * x, 0) - 3.0 * x) / (SINH(3.0 * tau0, 0) - 3.0 * tau0) - rval;

	//return ((sinh(3.0 * x) - 3.0 * x) / (sinh(3.0 * tau0) - 3.0 * tau0) - rval);
}

double tauPrime(const double &x, const double &tau0)
{
	return 6.0 * POW2(SINH(1.5 * x, 0), 0) / (SINH(3.0 * tau0, 0) - 3.0 * tau0);

	//return (6.0 * sinh(1.5 * x) / (sinh(3.0 * tau0) - 3.0 * tau0));
}

double phi4D(const double &x, const double &rval)
{
	return x * SIN(2.0 * x, 0) / TWO_PI - rval;	

	//return (((x - sin(x) * cos(x)) / M_PI) - rval);
}

double phiPrime4D(const double &x)
{
	return POW2(SIN(x, 0), 0) / HALF_PI;

	//return ((2 / M_PI) * sin(x) * sin(x));
}

//Math Function for Gauss Hypergeometric Function
double _2F1_z(const double &tau)
{
	return POW2(COSH(1.5 * tau, 0), 0);

	//return cosh(1.5 * tau) * cosh(1.5 * tau);
}

//De Sitter Spatial Lengths

//X1 Coordinate of de Sitter Metric
float X1(const float &phi)
{
	return COS(phi, 0);

	//return cosf(phi);
}

//X2 Coordinate of de Sitter Metric
float X2(const float &phi, const float &chi)
{
	return SIN(phi, 0) * COS(chi, 0);

	//return (sinf(phi) * cosf(chi));
}

//X3 Coordinate of de Sitter Metric
float X3(const float &phi, const float &chi, const float &theta)
{
	return SIN(phi, 0) * SIN(chi, 0) * COS(theta, 0);

	//return (sinf(phi) * sinf(chi) * cosf(theta));
}

//X4 Coordinate of de Sitter Metric
float X4(const float &phi, const float &chi, const float &theta)
{
	return SIN(phi, 0) * SIN(chi, 0) * SIN(theta, 0);

	//return (sinf(phi) * sinf(chi) * sinf(theta));
}

//Temporal Transformations

//Conformal to Rescaled Time
float etaToT(const float eta, const double a)
{
	return ACOSH(1.0 / COS(eta, 0), 0, HIGH_PRECISION) * a;

	//return (acoshf(1.0 / cosf(eta)) * a);
}

//Rescaled to Conformal Time
float tToEta(const float t, const double a)
{
	return ACOS(1.0 / COSH(t / a, 0), 0, HIGH_PRECISION);

	//return (acosf(1.0 / coshf(t / a)));
}
