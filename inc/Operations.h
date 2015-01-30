#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include "Causet.h"
#include "Subroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Newton-Raphson Kernels

inline double eta02D(const double &x, const int &N_tar, const float &k_tar)
{
	double _tanx = TAN(x, APPROX ? FAST : STL);
	return ((x / _tanx - LOG(COS(x, APPROX ? FAST : STL), APPROX ? FAST : STL) - 1.0) / (_tanx * HALF_PI)) - (k_tar / N_tar);
}

inline double eta0Prime2D(const double &x)
{
	double _cotx = 1.0 / TAN(x, APPROX ? FAST : STL);
	double _cscx2 = 1.0 / POW2(SIN(x, APPROX ? FAST : STL), EXACT);
	double _lnsecx = -1.0 * LOG(COS(x, APPROX ? FAST : STL), APPROX ? FAST : STL);

	return (_cotx * (_cotx - x * _cscx2) + 1 - _cscx2 * (_lnsecx + x * _cotx - 1)) / HALF_PI;
}

inline double zeta4D(const double &x, const int &N_tar, const float &k_tar)
{
	double _tanx = TAN(x, APPROX ? FAST : STL);
	double _cscx2 = 1.0 / POW2(SIN(x, APPROX ? FAST : STL), EXACT);
	double _lncscx = 0.5 * LOG(_cscx2, APPROX ? FAST : STL);

	return _tanx * (12.0 * (x * _tanx + _lncscx) + (6.0 * _lncscx - 5.0) * _cscx2 - 7.0) / (3.0 * HALF_PI * POW2(2.0f + _cscx2, EXACT)) - (k_tar / N_tar);
}

inline double zetaPrime4D(const double &x)
{
	double _cscx2 = 1.0 / POW2(SIN(x, APPROX ? FAST : STL), EXACT);
	double _sinx3 = POW3(SIN(x, APPROX ? FAST : STL), EXACT);
	double _sinx4 = POW(SIN(x, APPROX ? FAST : STL), 4.0, APPROX ? FAST : STL);
	double _cosx3 = POW3(COS(x, APPROX ? FAST : STL), EXACT);
	double _lncscx = -1.0 * LOG(SIN(x, APPROX ? FAST : STL), APPROX ? FAST : STL);

	return (3.0 * (COS(5.0 * x, APPROX ? FAST : STL) - 32.0 * (M_PI - 2.0 * x) * _sinx3) + COS(x, APPROX ? FAST : STL) * (84.0 - 72.0 * _lncscx) + COS(3.0 * x, APPROX ? FAST : STL) * (24.0 * _lncscx - 31.0)) / (-4.0 * M_PI * _sinx4 * _cosx3 * POW3((2.0 + _cscx2), EXACT));
}

inline double tau0Compact(const double &x, const int &N_tar, const double &alpha, const double &delta, const double &a)
{
	return SINH(3.0 * x, APPROX ? FAST : STL) - 3.0 * (x + static_cast<double>(N_tar) / (POW2(M_PI, EXACT) * delta * a * POW3(alpha, EXACT)));
}

inline double tau0Flat(const double &x, const int &N_tar, const double &alpha, const double &delta, const double &a, const double &chi_max)
{
	return SINH(3.0 * x, APPROX ? FAST : STL) - 3.0 * (x + static_cast<double>(N_tar) / (1.5 * M_PI * delta * a * POW3(alpha * chi_max, EXACT)));
}

inline double tau0Prime(const double &x)
{
	return 3.0 * (COSH(3.0 * x, APPROX ? FAST : STL) - 1.0);
}

inline double tau4D(const double &x, const double &zeta, const double &rval)
{
	double _coshx2 = POW2(COSH(x, APPROX ? FAST : STL), EXACT);
	double _sinz2 = POW2(SIN(zeta, APPROX ? FAST : STL), EXACT);

	return (2.0 + _coshx2) * SINH(x, APPROX ? FAST : STL) * TAN(zeta, APPROX ? FAST : STL) * _sinz2 / (2.0 * _sinz2 + 1) - rval;
}

inline double tauPrime4D(const double &x, const double &zeta)
{
	double _coshx2 = POW2(COSH(x, APPROX ? FAST : STL), EXACT);
	double _coshx4 = POW2(_coshx2, EXACT);
	double _sinz2 = POW2(SIN(zeta, APPROX ? FAST : STL), EXACT);

	return 3.0 * _coshx4 * TAN(zeta, APPROX ? FAST : STL) * _sinz2 / (2.0 * _sinz2 + 1);
}

inline double tauUniverse(const double &x, const double &tau0, const double &rval)
{
	return (SINH(3.0 * x, APPROX ? FAST : STL) - 3.0 * x) / (SINH(3.0 * tau0, APPROX ? FAST : STL) - 3.0 * tau0) - rval;
}

inline double tauPrimeUniverse(const double &x, const double &tau0)
{
	return 6.0 * POW2(SINH(1.5 * x, APPROX ? FAST : STL), EXACT) / (SINH(3.0 * tau0, APPROX ? FAST : STL) - 3.0 * tau0);
}

inline double theta1_4D(const double &x, const double &rval)
{
	return (2.0 * x - SIN(2.0 * x, APPROX ? FAST : STL)) / TWO_PI - rval;
}

inline double theta1_Prime4D(const double &x)
{
	return POW2(SIN(x, APPROX ? FAST : STL), EXACT) / HALF_PI;
}

//Returns zeta Residual
//Used in 1+1 and 3+1 Causets
inline double solveZeta(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG) {
		assert (p2 != NULL);
		assert (p3 != NULL);
		assert (p2[0] > 0.0f);			//k_tar
		assert (p3[0] > 0);			//N_tar
		assert (p3[1] == 1 || p3[1] == 3);	//dim
	}

	return ((p3[1] == 1) ?
		-1.0 * eta02D(x, p3[0], p2[0]) / eta0Prime2D(x) :
		-1.0 * zeta4D(x, p3[0], p2[0]) / zetaPrime4D(x));
}

//Returns tau0 Residual
//Used in Universe Causet
inline double solveTau0Compact(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG) {
		assert (p1 != NULL);
		assert (p3 != NULL);
		assert (p1[0] > 0.0);	//alpha
		assert (p1[1] > 0.0);	//delta
		assert (p1[2] > 0.0);	//a
		assert (p3[0] > 0);	//N_tar
	}

	return (-1.0 * tau0Compact(x, p3[0], p1[0], p1[1], p1[2]) / tau0Prime(x));
}

inline double solveTau0Flat(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG) {
		assert (p1 != NULL);
		assert (p3 != NULL);
		assert (p1[0] > 0.0);	//alpha
		assert (p1[1] > 0.0);	//delta
		assert (p1[2] > 0.0);	//a
		assert (p1[3] > 0.0);	//chi_max
		assert (p3[0] > 0);	//N_tar
	}

	return (-1.0 * tau0Flat(x, p3[0], p1[0], p1[1], p1[2], p1[3]) / tau0Prime(x));
}

//Returns tau Residual
//Used in 3+1 Causet
inline double solveTau(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG) {
		assert (p1 != NULL);
		assert (p1[0] > 0.0 && p1[0] < HALF_PI);	//zeta
		assert (p1[1] > 0.0 && p1[1] < 1.0);		//rval
	}

	return (-1.0 * tau4D(x, p1[0], p1[1]) / tauPrime4D(x, p1[0]));
}

//Returns tau Residual
//Used in Universe Causet
inline double solveTauUniverse(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG) {
		assert (p1 != NULL);
		assert (p1[0] > 0.0);			//tau0
		assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	}

	return (-1.0 * tauUniverse(x, p1[0], p1[1]) / tauPrimeUniverse(x, p1[0]));
}

//Returns tau Residual in Bisection Algorithm
//Used in Universe Causet
inline double solveTauUnivBisec(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG) {
		assert (p1 != NULL);
		assert (p1[0] > 0.0);			//tau0
		assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	}

	return tauUniverse(x, p1[0], p1[1]);
}

//Returns theta1 Residual
//Used in 3+1 and Universe Causets
inline double solveTheta1(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG) {
		assert (p1 != NULL);
		assert (p1[0] > 0.0 && p1[0] < 1.0);	//rval
	}

	return (-1.0 * theta1_4D(x, p1[0]) / theta1_Prime4D(x));
}

//Functions used for solving constraints in NetworkCreator.cu/initVars()

inline double solveDeltaCompact(const int &N_tar, const double &a, const double &tau0, const double &alpha)
{
	if (DEBUG) {
		assert (N_tar > 0);
		assert (a > 0.0);
		assert (tau0 > 0.0);
		assert (alpha > 0.0);
	}

	double delta;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		delta = exp(LOG(6.0 * N_tar / (POW2(M_PI, EXACT) * a * POW3(alpha, EXACT)), STL) - 3.0 * tau0);
	else
		delta = 3.0 * N_tar / (POW2(M_PI, EXACT) * a * POW3(alpha, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));

	if (DEBUG)
		assert (delta > 0.0);

	return delta;
}

inline double solveDeltaFlat(const int &N_tar, const double &a, const double &chi_max, const double &tau0, const double &alpha)
{
	if (DEBUG) {
		assert (N_tar > 0);
		assert (a > 0.0);
		assert (chi_max > 0.0);
		assert (tau0 > 0.0);
		assert (alpha > 0.0);
	}

	double delta;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		delta = exp(LOG(9.0 * N_tar / (M_PI * a * POW3(alpha * chi_max, EXACT)), STL) - 3.0 * tau0);
	else
		delta = 9.0 * N_tar / (2.0 * M_PI * a * POW3(alpha * chi_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));

	if (DEBUG)
		assert (delta > 0.0);

	return delta;
}

inline int solveNtarCompact(const double &a, const double &tau0, const double &alpha, const double &delta)
{
	if (DEBUG) {
		assert (a > 0.0);
		assert (tau0 > 0.0);
		assert (alpha > 0.0);
		assert (delta > 0.0);
	}

	int N_tar;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		N_tar = static_cast<int>(exp(LOG(6.0 / (POW2(M_PI, EXACT) * delta * a * POW3(alpha, EXACT)), EXACT) - 3.0 * tau0));
	else
		N_tar = static_cast<int>(POW2(M_PI, EXACT) * delta * a * POW3(alpha, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0) / 3.0);

	if (DEBUG)
		assert (N_tar > 0);

	return N_tar;
}

inline int solveNtarFlat(const double &a, const double &chi_max, const double &tau0, const double &alpha, const double &delta)
{
	if (DEBUG) {
		assert (a > 0.0);
		assert (chi_max > 0.0);
		assert (tau0 > 0.0);
		assert (alpha > 0.0);
		assert (delta > 0.0);
	}

	int N_tar;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		N_tar = static_cast<int>(exp(LOG(9.0 / (M_PI * delta * a * POW3(alpha * chi_max, EXACT)), EXACT) - 3.0 * tau0));
	else
		N_tar = static_cast<int>(2.0 * M_PI * delta * a * POW3(alpha * chi_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0) / 9.0);

	if (DEBUG)
		assert (N_tar > 0);

	return N_tar;
}

inline double solveAlphaCompact(const int &N_tar, const double &a, const double &tau0, const double &delta)
{
	if (DEBUG) {
		assert (N_tar > 0);
		assert (a > 0.0);
		assert (tau0 > 0.0);
		assert (delta > 0.0);
	}

	double alpha;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		alpha = exp(LOG(6.0 * N_tar / (POW2(M_PI, EXACT) * delta * a), STL) / 3.0 - tau0);
	else
		alpha = POW(3.0 * N_tar / (POW2(M_PI, EXACT) * delta * a * (SINH(3.0 * tau0, STL) - 3.0 * tau0)), (1.0 / 3.0), STL);

	if (DEBUG)
		assert (alpha > 0.0);

	return alpha;
}

inline double solveAlphaFlat(const int &N_tar, const double &a, const double &chi_max, const double &tau0, const double &delta)
{
	if (DEBUG) {
		assert (N_tar > 0);
		assert (a > 0.0);
		assert (chi_max > 0.0);
		assert (tau0 > 0.0);
		assert (delta > 0.0);
	}

	double alpha;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		alpha = exp(LOG(9.0 * N_tar / (M_PI * delta * a), STL) / 3.0 - tau0);
	else
		alpha = POW(9.0 * N_tar / (2.0 * M_PI * delta * a * (SINH(3.0 * tau0, STL) - 3.0 * tau0)), (1.0 / 3.0), STL);

	if (DEBUG)
		assert (alpha > 0.0);

	return alpha;
}

//Math Functions for Gauss Hypergeometric Function

//This is used to solve for a more exact solution than the one provided
//by numerical integration using the tauToEtaUniverse function
inline double _2F1_tau(const double &tau, void * const param)
{
	if (DEBUG)
		assert (tau > 0.0);

	return 1.0 / POW2(COSH(1.5 * tau, APPROX ? FAST : STL), EXACT);
}

//This is used to evaluate xi(r) in the rescaledDegreeUniverse
//function for r > 1
inline double _2F1_r(const double &r, void * const param)
{
	return -1.0 / POW3(r, EXACT);
}

//De Sitter Spatial Lengths

//X1 Coordinate of de Sitter Metric
inline float X1_SPH(const float &theta1)
{
	return static_cast<float>(COS(theta1, APPROX ? FAST : STL));
}

//X2 Coordinate of de Sitter Metric
inline float X2_SPH(const float &theta1, const float &theta2)
{
	return static_cast<float>(SIN(theta1, APPROX ? FAST : STL) * COS(theta2, APPROX ? FAST : STL));
}

//X3 Coordinate of de Sitter Metric
inline float X3_SPH(const float &theta1, const float &theta2, const float &theta3)
{
	return static_cast<float>(SIN(theta1, APPROX ? FAST : STL) * SIN(theta2, APPROX ? FAST : STL) * COS(theta3, APPROX ? FAST : STL));
}

//X4 Coordinate of de Sitter Metric
inline float X4_SPH(const float &theta1, const float &theta2, const float &theta3)
{
	return static_cast<float>(SIN(theta1, APPROX ? FAST : STL) * SIN(theta2, APPROX ? FAST : STL) * SIN(theta3, APPROX ? FAST : STL));
}

//X Coordinate from Spherical Basis
inline float X_FLAT(const float &theta1, const float &theta2, const float &theta3)
{
	return static_cast<float>(theta1 * SIN(theta2, APPROX ? FAST : STL) * COS(theta3, APPROX ? FAST : STL));
}

//Y Coordinate from Spherical Basis
inline float Y_FLAT(const float &theta1, const float &theta2, const float &theta3)
{
	return static_cast<float>(theta1 * SIN(theta2, APPROX ? FAST : STL) * SIN(theta3, APPROX ? FAST : STL));
}

//Z Coordinate from Spherical Basis
inline float Z_FLAT(const float &theta1, const float &theta2)
{
	return static_cast<float>(theta1 * COS(theta2, APPROX ? FAST : STL));
}

//Spherical Inner Product
//Returns angle between two points on unit sphere
inline float sphProduct_v1(const float4 &sc0, const float4 &sc1)
{
	return X1_SPH(sc0.x) * X1_SPH(sc1.x) +
	       X2_SPH(sc0.x, sc0.y) * X2_SPH(sc1.x, sc1.y) +
	       X3_SPH(sc0.x, sc0.y, sc0.z) * X3_SPH(sc1.x, sc1.y, sc1.z) +
	       X4_SPH(sc0.x, sc0.y, sc0.z) * X4_SPH(sc1.x, sc1.y, sc1.z);
}

//Factored form, fewer FLOPs than v1
inline float sphProduct_v2(const float4 &sc0, const float4 &sc1)
{
	return cosf(sc0.x) * cosf(sc1.x) +
	       sinf(sc0.x) * sinf(sc1.x) * (cosf(sc0.y) * cosf(sc1.y) + 
	       sinf(sc0.y) * sinf(sc1.y) * cosf(sc0.z - sc1.z));
}

//Flat Inner Product
//Returns distance ***SQUARED***
inline float flatProduct_v1(const float4 &sc0, const float4 &sc1)
{
	return POW2(X_FLAT(sc0.x, sc0.y, sc0.z) - X_FLAT(sc1.x, sc1.y, sc1.z), EXACT) +
	       POW2(Y_FLAT(sc0.x, sc0.y, sc0.z) - Y_FLAT(sc1.x, sc1.y, sc1.z), EXACT) +
	       POW2(Z_FLAT(sc0.x, sc0.y) - Z_FLAT(sc1.x, sc1.y), EXACT);
}

//Factored form, fewer FLOPS than v1
inline float flatProduct_v2(const float4 &sc0, const float4 &sc1)
{
	return POW2(sc0.x, EXACT) + POW2(sc1.x, EXACT) -
	       2.0f * sc0.x * sc1.x * (cosf(sc0.y) * cosf(sc1.y) +
	       sinf(sc0.y) * sinf(sc1.y) * cosf(sc0.z - sc1.z));
}

//Temporal Transformations

//Conformal to Rescaled Time
inline double etaToTau(const double eta)
{
	if (DEBUG)
		assert (eta > 0.0 && eta < HALF_PI);

	return ACOSH(1.0 / COS(eta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Rescaled to Conformal Time
inline double tauToEta(const double tau)
{
	if (DEBUG)
		assert (tau > 0.0);

	return ACOS(1.0 / COSH(tau, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Minkowski to Conformal Time (Universe)

//For use with GNU Scientific Library
inline double tToEtaUniverse(double t, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (t > 0.0);
	}

	//Identify params[0] with 'a'
	double a = ((double*)params)[0];

	return POW(SINH(1.5 * t / a, APPROX ? FAST : STL), (-2.0 / 3.0), APPROX ? FAST : STL);
}

//Exact Solution - Power Series Approximation
inline double tauToEtaUniverseExact(const double &tau, const double &a, const double &alpha)
{
	if (DEBUG) {
		assert (tau > 0.0);
		assert (a > 0.0);
		assert (alpha > 0.0);
	}

	double z = _2F1_tau(tau, NULL);
	double eta, f, err;
	int nterms = 50;

	_2F1(&_2F1_tau, tau, NULL, 1.0 / 3.0, 5.0 / 6.0, 4.0 / 3.0, &f, &err, &nterms);

	eta = 3.0 * GAMMA(-1.0 / 3.0, STL) * POW(z, 1.0 / 3.0, APPROX ? FAST : STL) * f;
	eta += 4.0 * SQRT(3.0, STL) * POW(M_PI, 1.5, APPROX ? FAST : STL) / GAMMA(5.0 / 6.0, STL);
	eta *= a / (9.0 * alpha * GAMMA(2.0 / 3.0, STL));

	if (DEBUG)
		assert (eta > 0.0);

	return eta;
}

//Gives Input to Lookup Table
inline double etaToTauUniverse(const double &eta, const double &a, const double &alpha)
{
	if (DEBUG) {
		assert (eta > 0.0);
		assert (a > 0.0);
		assert (alpha > 0.0);
	}

	double g = 9.0 * GAMMA(2.0 / 3.0, STL) * alpha * eta / a;
	g -= 4.0 * SQRT(3.0, APPROX ? BITWISE : STL) * POW(M_PI, 1.5, STL) / GAMMA(5.0 / 6.0, STL);
	g /= 3.0 * GAMMA(-1.0 / 3.0, STL);

	if (DEBUG)
		assert (g > 0.0);

	return g;
}

//Rescaled Average Degree in Universe Causet

//Approximates (108) in [2]
inline double xi(double &r)
{
	double _xi = 0.0;
	double err = 0.0;
	double f;
	int nterms = 10;

	if (ABS(r - 1.0, STL) < 0.05)
		nterms = 20;

	if (r < 1.0) {
		//Since 1/f(x) = f(1/x) we can use _r
		double _r = 1.0 / r;
		_2F1(&_2F1_r, _r, NULL, 1.0 / 6.0, 0.5, 7.0 / 6.0, &f, &err, &nterms);
		_xi = 2.0 * SQRT(r, STL) * f;
	} else {
		_2F1(&_2F1_r, r, NULL, 1.0 / 3.0, 0.5, 4.0 / 3.0, &f, &err, &nterms);
		_xi = SQRT(4.0 / M_PI, STL) * GAMMA(7.0 / 6.0, STL) * GAMMA(1.0 / 3.0, STL) - f / r;
	}

	return _xi;
}

//Note to get the rescaled averge degree this result must still be
//multiplied by 8pi/(sinh(3tau0)-3tau0)
inline double rescaledDegreeUniverse(int dim, double x[], double *params)
{
	if (DEBUG) {
		assert (dim > 0);
		assert (x[0] > 0.0);
		assert (x[1] > 0.0);
	}

	//Identify x[0] with x coordinate
	//Identify x[1] with r coordinate

	double z;

	z = POW3(ABS(xi(x[0]) - xi(x[1]), STL), EXACT) * POW2(x[0], EXACT) * POW3(x[1], EXACT) * SQRT(x[1], STL);
	z /= (SQRT(1.0 + 1.0 / POW3(x[0], EXACT), STL) * SQRT(1.0 + POW3(x[1], EXACT), STL));

	if (DEBUG)
		assert (z > 0.0);

	return z;
}

//Average Degree in Universe Causet (not rescaled)

//Gives rescaled scale factor as a function of eta
inline double rescaledScaleFactor(double *table, double size, double eta, double a, double alpha)
{
	if (DEBUG) {
		assert (table != NULL);
		assert (size > 0.0);
		assert (eta > 0.0);
		assert (a > 0.0);
		assert (alpha > 0.0);
	}

	double g = etaToTauUniverse(eta, a, alpha);
	if (DEBUG) 
		assert (g > 0.0);

	long l_size = static_cast<long>(size);

	double tau = lookupValue(table, l_size, NULL, &g, false);

	if (tau != tau) {
		if (g > table[0])
			tau = table[1] / 2.0;
		else {
			printf("Value not found in ctuc_table.cset.bin:\n");
			printf("\tEta: %f\n", eta);
			printf("\tg:   %f\n", g);
			exit(0);
		}
	}
	
	if (DEBUG)
		assert (tau > 0.0);
		
	return POW(SINH(1.5 * tau, APPROX ? FAST : STL), 2.0 / 3.0, APPROX ? FAST : STL);
}

//Note to get the average degree this result must still be
//multipled by (4pi/3)*delta*alpha^4/psi
inline double averageDegreeUniverse(int dim, double x[], double *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (dim > 0);
		assert (x[0] > 0.0);
		assert (x[1] > 0.0);
		assert (params[0] > 0.0);
		assert (params[1] > 0.0);
		assert (params[2] > 0.0);
	}

	//Identify x[0] with eta'
	//Identify x[1] with eta''
	//Identify params[0] with a
	//Identify params[1] with alpha
	//Identify params[2] with size
	//Identify params[3] with table

	double z;

	z = POW3(ABS(x[0] - x[1], STL), EXACT);
	z *= POW2(POW2(rescaledScaleFactor(&params[3], params[2], x[0], params[0], params[1]), EXACT), EXACT);
	z *= POW2(POW2(rescaledScaleFactor(&params[3], params[2], x[1], params[0], params[1]), EXACT), EXACT);

	if (DEBUG)
		assert (z > 0.0);

	return z;
}

//For use with GNU Scientific Library
inline double psi(double eta, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (eta > 0.0);
	}

	//Identify params[0] with a
	//Identify params[1] with alpha
	//Identify params[2] with size
	//Identify params[3] with table
	
	return POW2(POW2(rescaledScaleFactor(&((double*)params)[3], ((double*)params)[2], eta, ((double*)params)[0], ((double*)params)[1]), EXACT), EXACT);
}

//For use with GNU Scientific Library
inline double degreeFieldTheory(double eta, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (eta > 0.0);
	}

	//Identify params[0] with eta_m
	//Identify params[1] with a
	//Identify params[2] with alpha
	//Identify params[3] with size
	//Identify params[4] with table
	
	return POW3(ABS(((double*)params)[0] - eta, STL), EXACT) * POW2(POW2(rescaledScaleFactor(&((double*)params)[4], ((double*)params)[3], eta, ((double*)params)[1], ((double*)params)[2]), EXACT), EXACT);
}

//Geodesic Distances

//Embedded Z1 Coordinate
//Used to calculate geodesic distances in universe
//For use with GNU Scientific Library
inline double embeddedZ1(double x, void *params)
{
	if (DEBUG)
		assert (params != NULL);

	//GSL_EmbeddedZ1_Parameters *p = (GSL_EmbeddedZ1_Parameters*)params;
	double *p = (double*)params;

	//double a = p->a;
	//double alpha = p->alpha;
	double a = p[0];
	double alpha = p[1];
	
	if (DEBUG) {
		assert (a > 0.0);
		assert (alpha > 0.0);
	}

	return SQRT(1.0 + POW2(a, EXACT) * x * POW2(alpha, EXACT) / (POW3(alpha, EXACT) + POW3(x, EXACT)), STL);
}

//Returns the embedded FLRW distance between two nodes
//Modify this to handle 3+1 DS without matter!
//O(xxx) Efficiency (revise this)
inline double distanceEmbFLRW(const float4 &node_a, const float &tau_a, const float4 &node_b, const float &tau_b, const int &dim, const Manifold &manifold, const double &a, const double &alpha, const bool &universe, const bool &compact)
{
	if (DEBUG) {
		assert (dim == 3);
		assert (manifold == DE_SITTER);
		assert (a > 0.0);
		if (universe)
			assert (alpha > 0.0);
		assert (compact);
	}

	//Check if they are the same node
	if (node_a.w == node_b.w && node_a.x == node_b.x && node_a.y == node_b.y && node_a.z == node_b.z)
		return 0.0;

	double z0_a, z0_b;
	double z1_a, z1_b;
	double inner_product_ab;
	double distance;

	if (universe) {
		IntData idata = IntData();
		idata.tol = 1e-5;

		double p[2];
		p[0] = a;
		p[1] = alpha;

		//Solve for z1 in Rotated Plane
		double power = 2.0 / 3.0;
		z1_a = alpha * POW(SINH(1.5 * tau_a, APPROX ? FAST : STL), power, APPROX ? FAST : STL);
		z1_b = alpha * POW(SINH(1.5 * tau_b, APPROX ? FAST : STL), power, APPROX ? FAST : STL);

		//Use Numerical Integration for z0
		idata.upper = z1_a;
		z0_a = integrate1D(&embeddedZ1, (void*)p, &idata, QNG);
		idata.upper = z1_b;
		z0_b = integrate1D(&embeddedZ1, (void*)p, &idata, QNG);
	} else {
		z0_a = a * SINH(tau_a, APPROX ? FAST : STL);
		z0_b = a * SINH(tau_b, APPROX ? FAST : STL);

		z1_a = a * COSH(tau_a, APPROX ? FAST : STL);
		z1_b = a * COSH(tau_b, APPROX ? FAST : STL);
	}

	if (DIST_V2)
		inner_product_ab = z1_a * z1_b * sphProduct_v2(node_a, node_b) - z0_a * z0_b;
	else
		inner_product_ab = z1_a * z1_b * sphProduct_v1(node_a, node_b) - z0_a * z0_b;

	if (inner_product_ab > 1.0)
		//Timelike
		distance = ACOSH(inner_product_ab, APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
	else if (inner_product_ab < -1.0)
		//Disconnected Regions
		//Negative sign indicates not timelike
		distance = -1.0 * INF;
	else
		//Spacelike
		//Negative sign indicates not timelike
		distance = -1.0 * ACOS(inner_product_ab, APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);

	return distance;
}

//Returns the hyperbolic distance between two nodes
//O(xxx) Efficiency (revise this)
inline double distanceH(const float2 &hc_a, const float2 &hc_b, const int &dim, const Manifold &manifold, const double &zeta)
{
	if (DEBUG) {
		assert (dim == 1);
		assert (manifold == HYPERBOLIC);
		assert (zeta != 0.0);
	}

	if (hc_a.x == hc_b.x && hc_a.y == hc_b.y)
		return 0.0f;

	double dtheta = M_PI - ABS(M_PI - ABS(hc_a.y - hc_b.y, STL), STL);
	double distance = ACOSH(COSH(zeta * hc_a.x, APPROX ? FAST : STL) * COSH(zeta * hc_b.x, APPROX ? FAST : STL) - SINH(zeta * hc_a.x, APPROX ? FAST : STL) * SINH(zeta * hc_b.x, APPROX ? FAST : STL) * COS(dtheta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION) / zeta;

	return distance;
}

#endif
