#include "Operations.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

//Returns zeta Residual
//Used in 1+1 and 3+1 Causets
double solveZeta(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		//No null pointers
		assert (p4 != NULL);	//k_tar
		assert (p5 != NULL);	//N_tar
		assert (p6 != NULL);	//dim

		//Variables in the correct ranges
		assert (*p4 > 0.0);
		assert (*p5 > 0);
		assert (*p6 == 1 || *p6 == 3);
	}

	return ((*p6 == 1) ?
		-1.0 * eta02D(x, *p5, *p4) / eta0Prime2D(x) : 
		-1.0 * zeta4D(x, *p5, *p4) / zetaPrime4D(x));
}

//Returns tau0 Residual
//Used in Universe Causet
double solveTau0(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		//No null pointers
		assert (p1 != NULL);	//alpha
		assert (p2 != NULL);	//delta
		assert (p5 != NULL);	//N_tar

		//Variables in the correct ranges
		assert (*p1 > 0.0);
		assert (*p2 > 0);
		assert (*p5 > 0);
	}

	return (-1.0 * tau0(x, *p5, *p1, *p2) / tau0Prime(x));
}

//Returns tau Residual
//Used in 3+1 Causet
double solveTau(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		//No null pointers
		assert (p1 != NULL);	//zeta
		assert (p3 != NULL);	//rval

		//Variables in the correct ranges
		assert (*p1 > 0.0 && *p1 < HALF_PI);
		assert (*p3 > 0.0 && *p3 < 1.0);
	}

	return (-1.0 * tau4D(x, *p1, *p3) / tauPrime4D(x, *p1));
}

//Returns tau Residual
//Used in Universe Causet
double solveTauUniverse(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		//No null pointers
		assert (p1 != NULL);	//tau0
		assert (p2 != NULL);	//rval

		//Variables in the correct ranges
		assert (*p1 > 0.0);
		assert (*p2 > 0.0 && *p2 < 1.0);
	}

	return (-1.0 * tauUniverse(x, *p1, *p2) / tauPrimeUniverse(x, *p1));
}

//Returns phi Residual
//Used in 3+1 and Universe Causets
double solvePhi(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		//No null pointers
		assert (p1 != NULL);	//rval

		//Variables in correct ranges
		assert (*p1 > 0.0 && *p1 < 1.0);
	}

	return (-1.0 * phi4D(x, *p1) / phiPrime4D(x));
}

double eta02D(const double &x, const int &N_tar, const float &k_tar)
{
	double _tanx = TAN(static_cast<float>(x), APPROX ? FAST : STL);
	return ((x / _tanx - LOG(COS(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL) - 1.0) / (_tanx * HALF_PI)) - (k_tar / N_tar);
}

double eta0Prime2D(const double &x)
{
	double _cotx = 1.0 / TAN(static_cast<float>(x), APPROX ? FAST : STL);
	double _cscx2 = 1.0 / POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _lnsecx = -1.0 * LOG(COS(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL);

	return (_cotx * (_cotx - x * _cscx2) + 1 - _cscx2 * (_lnsecx + x * _cotx - 1)) / HALF_PI;
}

double zeta4D(const double &x, const int &N_tar, const float &k_tar)
{
	double _tanx = TAN(static_cast<float>(x), APPROX ? FAST : STL);
	double _cscx2 = 1.0 / POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _lncscx = 0.5 * LOG(_cscx2, APPROX ? FAST : STL);

	return _tanx * (12.0 * (x * _tanx + _lncscx) + (6.0 * _lncscx - 5.0) * _cscx2 - 7.0) / (3.0 * HALF_PI * POW2(2.0f + _cscx2, EXACT)) - (k_tar / N_tar);
}

double zetaPrime4D(const double &x)
{
	double _cscx2 = 1.0 / POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _sinx3 = POW3(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _sinx4 = POW(SIN(static_cast<float>(x), APPROX ? FAST : STL), 4.0, APPROX ? FAST : STL);
	double _cosx3 = POW3(COS(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _lncscx = -1.0 * LOG(SIN(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL);

	return (3.0 * (COS(5.0f * x, APPROX ? FAST : STL) - 32.0 * (M_PI - 2.0 * x) * _sinx3) + COS(static_cast<float>(x), APPROX ? FAST : STL) * (84.0 - 72.0 * _lncscx) + COS(3.0 * static_cast<float>(x), APPROX ? FAST : STL) * (24.0 * _lncscx - 31.0)) / (-4.0 * M_PI * _sinx4 * _cosx3 * POW3((2.0f + _cscx2), EXACT));
}

double tau0(const double &x, const int &N_tar, const double &alpha, const double &delta)
{
	return SINH(3.0f * x, APPROX ? FAST : STL) - 3.0 * (x + static_cast<double>(N_tar) / (POW2(static_cast<float>(M_PI), EXACT) * delta * POW3(static_cast<float>(alpha), EXACT)));
}

double tau0Prime(const double &x)
{
	return 3.0 * (COSH(3.0 * static_cast<float>(x), APPROX ? FAST : STL) - 1.0);
}

double tau4D(const double &x, const double &zeta, const double &rval)
{
	double _coshx2 = POW2(COSH(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _sinz2 = POW2(SIN(static_cast<float>(zeta), APPROX ? FAST : STL), EXACT);

	return (2.0 + _coshx2) * SINH(static_cast<float>(x), APPROX ? FAST : STL) * TAN(static_cast<float>(zeta), APPROX ? FAST : STL) * _sinz2 / (2.0 * _sinz2 + 1) - rval;
}

double tauPrime4D(const double &x, const double &zeta)
{
	double _coshx2 = POW2(COSH(static_cast<float>(x), APPROX ? FAST : STL), EXACT); 
	double _coshx4 = POW2(static_cast<float>(_coshx2), EXACT);
	double _sinz2 = POW2(SIN(static_cast<float>(zeta), APPROX ? FAST : STL), EXACT);

	return 3.0 * _coshx4 * TAN(static_cast<float>(zeta), APPROX ? FAST : STL) * _sinz2 / (2.0 * _sinz2 + 1);
}

double tauUniverse(const double &x, const double &tau0, const double &rval)
{
	return (SINH(3.0f * x, APPROX ? FAST : STL) - 3.0 * x) / (SINH(3.0f * tau0, APPROX ? FAST : STL) - 3.0 * tau0) - rval;
}

double tauPrimeUniverse(const double &x, const double &tau0)
{
	return 6.0 * POW2(SINH(1.5f * x, APPROX ? FAST : STL), EXACT) / (SINH(3.0f * tau0, APPROX ? FAST : STL) - 3.0 * tau0);
}

double phi4D(const double &x, const double &rval)
{
	return (2.0 * x - SIN(2.0f * x, APPROX ? FAST : STL)) / TWO_PI - rval;
}

double phiPrime4D(const double &x)
{
	return POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT) / HALF_PI;
}

//Math Functions for Gauss Hypergeometric Function

//This is used to solve for a more exact solution than the one provided
//by numerical integration using the tauToEtaUniverse function
float _2F1_tau(const float &tau, void * const param)
{
	return 1.0f / POW2(COSH(1.5f * tau, APPROX ? FAST : STL), EXACT);
}

//This is used to evaluate xi(r) in the rescaledDegreeUniverse
//function for r > 1
float _2F1_r(const float &r, void * const param)
{
	return -1.0f / POW3(r, EXACT);
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

//Conformal to Rescaled Time
float etaToTau(const float eta)
{
	return ACOSH(1.0 / COS(eta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Rescaled to Conformal Time
float tauToEta(const float tau)
{
	return ACOS(1.0 / COSH(tau, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Minkowski to Conformal Time (Universe)
//For use with GNU Scientific Library
double tauToEtaUniverse(double tau, void *params)
{
	return POW(SINH(1.5f * tau, APPROX ? FAST : STL), (-2.0 / 3.0), APPROX ? FAST : STL);
}

float tauToEtaUniverseExact(const float &tau, const float &a, const float &alpha)
{
	float z = _2F1_tau(tau, NULL);
	float eta, f, err;
	int nterms = 50;

	_2F1(&_2F1_tau, tau, NULL, 1.0f / 3.0f, 5.0f / 6.0f, 4.0f / 3.0f, &f, &err, &nterms);
	
	if (DEBUG) assert (ABS(err, STL) < 1e-4);

	eta = 3.0f * GAMMA(-1.0f / 3.0f, STL) * POW(z, 1.0f / 3.0f, APPROX ? FAST : STL) * f;
	eta += 4.0f * SQRT(3.0f, STL) * POW(M_PI, 1.5f, APPROX ? FAST : STL) / GAMMA(5.0f / 6.0f, STL);
	eta *= a / (9.0f * alpha * GAMMA(2.0f / 3.0f, STL));

	return eta;
}

//Rescaled Average Degree in Universe Causet

//Note to get the rescaled averge degree this result must still be 
//multiplied by 8pi/(sinh(3tau0)-3tau0)
double rescaledDegreeUniverse(int dim, double x[])
{
	//Identify x[0] with x coordinate
	//Identify x[1] with r coordinate

	double z;

	z = POW3(ABS(xi(x[0]) - xi(x[1]), STL), EXACT) * POW2(x[0], EXACT) * POW3(x[1], EXACT) * SQRT(x[1], STL);
	z /= (SQRT(1.0 + 1.0 / POW3(x[0], EXACT), STL) * SQRT(1.0 + POW3(x[1], EXACT), STL));

	return z;
}

//Approximates (108) in [2]
double xi(double &r)
{
	double _xi = 0.0;
	float err = 0.0f;
	float f;
	int nterms = 10;

	if (ABS(r - 1.0f, STL) < 0.05)
		nterms = 20;

	if (r < 1.0f) {
		//Since 1/f(x) = f(1/x) we can use _r
		double _r = 1.0f / r;
		_2F1(&_2F1_r, _r, NULL, 1.0f / 6.0f, 0.5f, 7.0f / 6.0f, &f, &err, &nterms);
		_xi = 2.0 * SQRT(r, STL) * f;
	} else {
		_2F1(&_2F1_r, r, NULL, 1.0f / 3.0f, 0.5f, 4.0f / 3.0f, &f, &err, &nterms);
		_xi = SQRT(4.0f / M_PI, STL) * GAMMA(7.0f / 6.0f, STL) * GAMMA(1.0f / 3.0f, STL) - f / r;
	}

	if (DEBUG) assert (ABS(err, STL) < 1e-4);

	return _xi;
}

