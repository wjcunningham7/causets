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
	double _tanx = static_cast<double>(TAN(static_cast<float>(x), APPROX ? FAST : STL));
	return ((x / _tanx - static_cast<double>(LOG(COS(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL)) - 1.0) / (_tanx * static_cast<double>(HALF_PI))) - (static_cast<double>(k_tar) / static_cast<double>(N_tar));
}

double eta0Prime2D(const double &x)
{
	double _cotx = 1.0 / static_cast<double>(TAN(static_cast<float>(x), APPROX ? FAST : STL));
	double _cscx2 = 1.0 / static_cast<double>(POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT));
	double _lnsecx = -1.0 * static_cast<double>(LOG(COS(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL));

	return (_cotx * (_cotx - x * _cscx2) + 1 - _cscx2 * (_lnsecx + x * _cotx - 1)) / static_cast<double>(HALF_PI);
}

double zeta4D(const double &x, const int &N_tar, const float &k_tar)
{
	double _tanx = static_cast<double>(TAN(static_cast<float>(x), APPROX ? FAST : STL));
	double _cscx2 = 1.0 / static_cast<double>(POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT));
	double _lncscx = -1.0 * static_cast<double>(LOG(SIN(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL));

	return _tanx * (12.0 * (x * _tanx + _lncscx) + (6.0 * _lncscx - 5.0) * _cscx2 - 7.0) / (3.0 * static_cast<double>(HALF_PI) * static_cast<double>(POW2(2.0 + static_cast<float>(_cscx2), EXACT))) - static_cast<double>(k_tar) / static_cast<double>(N_tar);
}

double zetaPrime4D(const double &x)
{
	double _cscx2 = 1.0 / static_cast<double>(POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT));
	double _sinx3 = static_cast<double>(POW3(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT));
	double _sinx4 = static_cast<double>(POW(SIN(static_cast<float>(x), APPROX ? FAST : STL), 4.0, APPROX ? FAST : STL));
	double _cosx3 = static_cast<double>(POW3(COS(static_cast<float>(x), APPROX ? FAST : STL), EXACT));
	double _lncscx = -1.0 * static_cast<double>(LOG(SIN(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL));

	return (3.0 * (static_cast<double>(COS(5.0 * static_cast<float>(x), APPROX ? FAST : STL)) - 32.0 * (static_cast<double>(M_PI) - 2.0 * x) * _sinx3) + static_cast<double>(COS(static_cast<float>(x), APPROX ? FAST : STL)) * (84.0 - 72.0 * _lncscx) + static_cast<double>(COS(3.0 * static_cast<float>(x), APPROX ? FAST : STL)) * (24.0 * _lncscx - 31.0)) / (-4.0 * static_cast<double>(M_PI) * _sinx4 * _cosx3 * static_cast<double>(POW3((2.0 + static_cast<float>(_cscx2)), EXACT)));
}

double tau0(const double &x, const int &N_tar, const double &alpha, const double &delta)
{
	return static_cast<double>(SINH(3.0 * static_cast<float>(x), APPROX ? FAST : STL)) - 3.0 * (x + static_cast<double>(N_tar) / (static_cast<double>(POW2(static_cast<float>(M_PI), EXACT)) * delta * static_cast<double>(POW3(static_cast<float>(alpha), EXACT))));
}

double tau0Prime(const double &x)
{
	return 3.0 * (static_cast<double>(COSH(3.0 * static_cast<float>(x), APPROX ? FAST : STL)) - 1.0);
}

double t4D(const double &x, const double &zeta, const double &a, const double &rval)
{
	double _coshx2 = static_cast<double>(POW(COSH(static_cast<float>(x / a), APPROX ? FAST : STL), 4.0, APPROX ? FAST : STL));
	double _sinz2 = static_cast<double>(POW2(SIN(static_cast<float>(zeta), APPROX ? FAST : STL), EXACT));

	return (2.0 + _coshx2) * static_cast<double>(SINH(static_cast<float>(x / a), APPROX ? FAST : STL)) * static_cast<double>(TAN(static_cast<float>(zeta), APPROX ? FAST : STL)) * _sinz2 / (2.0 * _sinz2 + 1) - rval;

	//return ((((2.0 + cosh(x / a) * cosh(x / a)) * sinh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta))) - rval);
}

double tPrime4D(const double &x, const double &zeta, const double &a)
{
	double _coshx2 = static_cast<double>(POW2(COSH(static_cast<float>(x / a), APPROX ? FAST : STL), EXACT)); 
	double _coshx4 = static_cast<double>(POW2(static_cast<float>(_coshx2), EXACT));
	double _sinz2 = static_cast<double>(POW2(SIN(static_cast<float>(zeta), APPROX ? FAST : STL), EXACT));

	return 3.0 * _coshx4 * static_cast<double>(TAN(static_cast<float>(zeta), APPROX ? FAST : STL)) * _sinz2 / (2.0 * _sinz2 + 1);

	//return ((3.0 * cosh(x / a) * cosh(x / a) * cosh(x / a) * cosh(x / a)) / ((2.0 + (1.0 / (sin(zeta) * sin(zeta)))) / tan(zeta)));
}

double tau(const double &x, const double &tau0, const double &rval)
{
	//return (static_cast<double>(SINH(3.0 * static_cast<float>(x), APPROX ? FAST : STL)) - 3.0 * x) / (static_cast<double>(SINH(3.0 * static_cast<float>(tau0), APPROX ? FAST : STL)) - 3.0 * tau0) - rval;

	return ((sinh(3.0 * x) - 3.0 * x) / (sinh(3.0 * tau0) - 3.0 * tau0) - rval);
}

double tauPrime(const double &x, const double &tau0)
{
	//return 6.0 * static_cast<double>(POW2(SINH(1.5 * static_cast<float>(x), APPROX ? FAST : STL), EXACT)) / (static_cast<double>(SINH(3.0 * static_cast<float>(tau0), APPROX ? FAST : STL)) - 3.0 * tau0);

	return (6.0 * sinh(1.5 * x) / (sinh(3.0 * tau0) - 3.0 * tau0));
}

double phi4D(const double &x, const double &rval)
{
	//return (2.0 * x - static_cast<double>(SIN(2.0 * static_cast<float>(x), APPROX ? FAST : STL))) / static_cast<double>(TWO_PI) - rval;

	return (((x - sin(x) * cos(x)) / M_PI) - rval);
}

double phiPrime4D(const double &x)
{
	//return static_cast<double>(POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT)) / static_cast<double>(HALF_PI);

	return ((2 / M_PI) * sin(x) * sin(x));
}

//Math Function for Gauss Hypergeometric Function
//This is used to solve for a more exact solution than the one provided
//by numerical integration using the tauToEtaUniverse function
float _2F1_z(const float &tau)
{
	return 1.0 / POW2(COSH(1.5 * tau, APPROX ? FAST : STL), EXACT);
}

//De Sitter Spatial Lengths

//X1 Coordinate of de Sitter Metric
float X1(const float &phi)
{
	return COS(phi, APPROX ? FAST : STL);
}

//X2 Coordinate of de Sitter Metric
float X2(const float &phi, const float &chi)
{
	return SIN(phi, APPROX ? FAST : STL) * COS(chi, APPROX ? FAST : STL);
}

//X3 Coordinate of de Sitter Metric
float X3(const float &phi, const float &chi, const float &theta)
{
	return SIN(phi, APPROX ? FAST : STL) * SIN(chi, APPROX ? FAST : STL) * COS(theta, APPROX ? FAST : STL);
}

//X4 Coordinate of de Sitter Metric
float X4(const float &phi, const float &chi, const float &theta)
{
	return SIN(phi, APPROX ? FAST : STL) * SIN(chi, APPROX ? FAST : STL) * SIN(theta, APPROX ? FAST : STL);
}

//Temporal Transformations

//Conformal to Minkowski Time
float etaToT(const float eta, const double a)
{
	return ACOSH(1.0 / COS(eta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION) * static_cast<float>(a);
	
	//return acoshf(1.0 / cosf(eta)) * a;
}

//Minkowski to Conformal Time
float tToEta(const float t, const double a)
{
	return ACOS(1.0 / COSH(t / static_cast<float>(a), APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);

	//return acosf(1.0 / coshf(t / a));
}

//Minkowski to Conformal Time (Universe)
//For use with GNU Scientific Library
double tToEtaUniverse(double t, void *params)
{
	return static_cast<double>(POW(SINH(1.5 * static_cast<float>(t), APPROX ? FAST : STL), (-2.0 / 3.0), APPROX ? FAST : STL));
}

float tToEtaUniverseExact(const float &tau, const float &a, const float &alpha, const float F_a, const float F_b, const float F_c)
{
	float g1 = GAMMA(F_c, STL) * GAMMA(F_b - F_a, STL) / (GAMMA(F_b, STL) * GAMMA(F_c - F_a, STL));
	float g2 = GAMMA(F_c, STL) * GAMMA(F_a - F_b, STL) / (GAMMA(F_a, STL) * GAMMA(F_c - F_b, STL));
	float g3 = GAMMA(F_c - F_a - F_b, STL) / GAMMA(F_b - F_a, STL);
	float z = 1.0 / _2F1_z(tau);
	float z_ab = POW(z, F_a + F_b, FAST);
	float eta = 0.0f;

	float f1, f2;
	bool success;

	success = _2F1(&_2F1_z, &f1, tau, F_a, 1.0f - F_c + F_a, 1.0f - F_b + F_a, 1e-3, VERY_HIGH_PRECISION);
	success &= _2F1(&_2F1_z, &f2, tau, F_b, 1.0f - F_c + F_b, 1.0 - F_a + F_b, 1e-3, VERY_HIGH_PRECISION);
	assert (success);

	eta = POW2(g3 * SQRT(static_cast<float>(M_PI), STL) * z_ab - 2.0f * z * (g1 * POW(z, F_b, APPROX ? FAST : STL) * COS(F_a * static_cast<float>(M_PI), APPROX ? FAST : STL) * f1 + g2 * POW(z, F_a, APPROX ? FAST : STL) * COS(F_b * static_cast<float>(M_PI), APPROX ? FAST : STL) * f2), EXACT);
	eta += POW2(g1 * POW(z, F_b, FAST) * SIN(F_a * static_cast<float>(M_PI), APPROX ? FAST : STL) * f1 + g2 * POW(z, F_a, APPROX ? FAST : STL) * SIN(F_b * static_cast<float>(M_PI), APPROX ? FAST : STL) * f2, EXACT);
	eta /= POW2(z_ab, EXACT);
	eta = SQRT(eta, STL) * a / (3.0f * alpha);

	return eta;
}
