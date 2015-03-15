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

/*inline double lambda4D(const double &x, const double &a, const float &tau1, const float &tau2, const float &omega12)
{
	double st1 = SINH(tau1, APPROX ? FAST : STL);
	double st2 = SINH(tau2, APPROX ? FAST : STL);

	double xi1 = SQRT(2.0 + x * (1.0 + COSH(2.0 * a * tau1, APPROX ? FAST : STL)), STL);
	double xi2 = SQRT(2.0 + x * (1.0 + COSH(2.0 * a * tau2, APPROX ? FAST : STL)), STL);

	return SQRT(2.0, STL) * (xi1 * st2 - xi2 * st1) / (xi1 * xi2 + 2.0 * st1 * st2) - TAN(omega12, APPROX ? FAST : STL);
}

inline double lambdaPrime4D(const double &x, const double &a, const float &tau1, const float &tau2)
{
	double st1 = SINH(tau1, APPROX ? FAST : STL);
	double st2 = SINH(tau2, APPROX ? FAST : STL);

	double xi1 = SQRT(2.0 + x * (1.0 + COSH(2.0 * a * tau1, APPROX ? FAST : STL)), STL);
	double xi2 = SQRT(2.0 + x * (1.0 + COSH(2.0 * a * tau2, APPROX ? FAST : STL)), STL);

	double xi1_2 = POW2(xi1, EXACT);
	double xi1_3 = POW3(xi1, EXACT);

	double xi2_2 = POW2(xi2, EXACT);
	double xi2_3 = POW3(xi2, EXACT);

	double psi1 = (SQRT(2.0, STL) / 2.0) * (1.0 + COSH(2.0 * a * tau1, APPROX ? FAST : STL));
	double psi2 = (SQRT(2.0, STL) / 2.0) * (1.0 + COSH(2.0 * a * tau2, APPROX ? FAST : STL));

	return (xi1_2 * st1 * (xi2_2 * xi2_3 + 2.0 * xi2_3 * psi1 * POW2(st2, EXACT)) - xi1_2 * xi1_3 * psi2 * st2 - 2.0 * xi1_3 * xi2_2 * psi2 * POW2(st1, EXACT) * st2) / (xi1_3 * xi2_3 * POW2(xi1 * xi2 + 2.0 * st1 * st2, EXACT));
}*/

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

//Returns lambda Residual
//Used in 3+1 Geodesic Calculations
/*inline double solveLambda4D(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG) {
		assert (p1 != NULL);
		assert (p2 != NULL);
		assert (p1[0] > 0.0);	//a
		assert (p2[0] > 0.0f);	//tau1
		assert (p2[1] > 0.0f);	//tau2
		assert (p2[2] > 0.0f);	//omega12
	}

	return (-1.0 * lambda4D(x, p1[0], p2[0], p2[1], p2[2]) / lambdaPrime4D(x, p1[0], p2[0], p2[1]));
}

//Returns lambda Residual in Bisection Algorithm
//Used in 3+1 Geodesic Calculations
inline double solveLambda4DBisec(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	if (DEBUG) {
		assert (p1 != NULL);
		assert (p2 != NULL);
		assert (p1[0] > 0.0);	//a
		assert (p2[0] > 0.0f);	//tau1
		assert (p2[1] > 0.0f);	//tau2
		assert (p2[2] > 0.0f);	//omega12
	}

	return lambda4D(x, p1[0], p2[0], p2[1], p2[1]);
}*/

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

//Rescaled Average Degree in Universe Causet (Compact)

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

//Average Degree in Universe Causet (not rescaled, compact)

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
			#ifdef MPI_ENABLED
			MPI_Abort(MPI_COMM_WORLD, 10);
			#else
			exit(10);
			#endif
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

//Average Degree in Non-Compact FLRW Spacetime

//Case 1: tau0 > 2.0 * chi_max
/*inline double averageDegreeFLRW_C1(const double &a, const double &chi_max, const double &tau0, const double &alpha, const double &delta)
{
	return (a * M_PI * POW3(alpha, EXACT) * delta * (24605 + 27 * POW3(chi_max, EXACT) * (2100 * tau0 + chi_max * (7700 + 55539 * POW2(POW2(chi_max, EXACT), EXACT) + 6048 * chi_max * tau0 - 63720 * POW3(chi_max, EXACT) * tau0 + 3360 * POW2(chi_max, EXACT) * (2 + 9 * POW2(tau0, EXACT)))) - 5040 * (2 + chi_max * (chi_max * (13 + 12 * chi_max * (chi_max - tau0)) - 4 * tau0)) * COSH(3 * chi_max, STL) + 35 * (-415 + 54 * chi_max * (14 * tau0 + chi_max * (-131 + chi_max * (chi_max * (-71 + 72 * chi_max * (2 * chi_max - tau0)) + 38 * tau0)))) * COSH(6 * chi_max, STL) - 156800 * COSH(3 * chi_max - 3 * tau0, STL) + 62720 * COSH(6 * chi_max - 3 * tau0, STL) + 94080 * COSH(3 * tau0, STL) - 1120 * COSH(6 * tau0, STL) + 2 * (280 * (2 + 9 * POW2(chi_max, EXACT)) * COSH(3 * chi_max - 6 * tau0, STL) + 3 * (210 * POW2(chi_max, EXACT) * (16 + 33 * POW2(chi_max, EXACT) + 27 * POW2(POW2(chi_max, EXACT), EXACT)) * COSH(6 * chi_max - 6 * tau0, STL) + 560 * (-2 * tau0 + chi_max * (11 + 36 * POW2(chi_max, EXACT) - 9 * chi_max * tau0)) * SINH(3 * chi_max, STL) - 35 * (21 * tau0 + 2 * chi_max * (-202 + 27 * chi_max * (7 * tau0 + chi_max * (-29 + 18 * POW2(chi_max, EXACT) - 3 * chi_max * tau0)))) * SINH(6 * chi_max, STL) + 280 * chi_max * COSH(3 * tau0, STL) * (45 * POW3(chi_max, EXACT) * COSH(3 * chi_max, STL) + 6 * chi_max * (40 + 45 * POW2(chi_max, EXACT) + 36 * POW2(POW2(chi_max, EXACT), EXACT) + (22 + 24 * POW2(chi_max, EXACT)) * COSH(6 * chi_max, STL) + 4 * chi_max * SINH(3 * chi_max, STL)) - 4 * (26 + 36 * POW2(chi_max, EXACT) + 27 * POW2(POW2(chi_max, EXACT), EXACT)) * SINH(6 * chi_max, STL)) - 28 * chi_max * COSH(6 * tau0, STL) * (135 * POW3(chi_max, EXACT) - 20 * (1 + 3 * POW2(chi_max, EXACT)) * SINH(3 * chi_max, STL) + (40 + 210 * POW2(chi_max, EXACT) + 189 * POW2(POW2(chi_max, EXACT), EXACT)) * SINH(6 * chi_max, STL)) + 8 * chi_max * (-5600 + 6 * POW2(chi_max, EXACT) * (-1610 + 27 * POW2(chi_max, EXACT) * (-49 + 60 * POW2(chi_max, EXACT) - 70 * chi_max * tau0)) + 3640 * COSH(6 * chi_max, STL) + 105 * chi_max * (-8 * chi_max * COSH(3 * chi_max, STL) + 12 * chi_max * (4 + 3 * POW2(chi_max, EXACT)) * COSH(6 * chi_max, STL) - 15 * POW2(chi_max, EXACT) * SINH(3 * chi_max, STL) - 4 * (11 + 12 * POW2(chi_max, EXACT)) * SINH(6 * chi_max, STL))) * SINH(3 * tau0, STL) + 28 * chi_max * (6 * POW2(chi_max, EXACT) * (10 + 27 * POW2(chi_max, EXACT)) - 20 * (1 + 3 * POW2(chi_max, EXACT)) * COSH(3 * chi_max, STL) + (40 + 210 * POW2(chi_max, EXACT) + 189 * POW2(POW2(chi_max, EXACT), EXACT)) * COSH(6 * chi_max, STL)) * SINH(6 * tau0, STL))))) / (204120 * POW3(chi_max, EXACT) * (-3 * tau0 + SINH(3 * tau0, STL)));
}

//Case 2: chi_max < tau0 < 2.0 * chi_max
inline double averageDegreeFLRW_C2(const double &a, const double &chi_max, const double &tau0, const double &alpha, const double &delta)
{
//	return -1 * (a * M_PI * POW3(alpha, EXACT) * delta * (-87325 + 9 * (40743 * POW2(POW2(POW2(chi_max, EXACT), EXACT), EXACT) - 171720 * POW2(POW2(chi_max, EXACT), EXACT) * POW3(chi_max, EXACT) * tau0 + 30240 * POW2(POW3(chi_max, EXACT), EXACT) * (2 + 9 * POW2(tau0, EXACT)) - 6048 * POW2(chi_max, EXACT) * POW3(chi_max, EXACT) * tau0 * (29 + 39 * POW2(tau0, EXACT)) + 420 * POW3(chi_max, EXACT) * tau0 * (145 + 240 * POW2(tau0, EXACT) + 108 * POW2(POW2(tau0, EXACT), EXACT)) + 120 * chi_max * tau0 * (560 + 840 * POW2(tau0, EXACT) + 378 * POW2(POW2(tau0, EXACT), EXACT) + 81 * POW2(POW3(tau0, EXACT), EXACT)) - 14 * POW2(tau0, EXACT) * (2240 + 1680 * POW2(tau0, EXACT) + 504 * POW2(POW2(tau0, EXACT), EXACT) + 81 * POW2(POW3(tau0, EXACT), EXACT)) + 420 * POW2(POW2(chi_max, EXACT), EXACT) * (-23 + 36 * POW2(tau0, EXACT) * (4 + 3 * POW2(tau0, EXACT))) - 420 * POW2(chi_max, EXACT) * (80 + 9 * POW2(tau0, EXACT) * (40 + 30 * POW2(tau0, EXACT) + 9 * POW2(POW2(tau0, EXACT), EXACT)))) + 5040 * (2 + chi_max * (chi_max * (13 + 12 * chi_max * (chi_max - tau0)) - 4 * tau0)) * COSH(3 * chi_max, STL) - 84035 * COSH(6 * chi_max - 6 * tau0, STL) + 156800 * COSH(3 * chi_max - 3 * tau0, STL) + 4480 * COSH(3 * tau0, STL) + 1120 * COSH(6 * tau0, STL) - 2 * (280 * (2 + 9 * POW2(chi_max, EXACT)) * COSH(3 * chi_max - 6 * tau0, STL) - 3 * (105 * (chi_max - tau0) * (18 * POW2(chi_max, EXACT) * POW3(chi_max, EXACT) + 523 * tau0 + 18 * POW2(POW2(chi_max, EXACT), EXACT) * tau0 + 45 * POW2(chi_max, EXACT) * tau0 * (17 + 4 * POW2(tau0, EXACT)) + 3 * POW3(tau0, EXACT) * (85 + 6 * POW2(tau0, EXACT)) - 9 * POW3(chi_max, EXACT) * (21 + 16 * POW2(tau0, EXACT)) - chi_max * (523 + 765 * POW2(tau0, EXACT) + 90 * POW2(POW2(tau0, EXACT), EXACT))) * COSH(6 * chi_max - 6 * tau0, STL) + 560 * (2 * tau0 + chi_max * (-11 + 9 * chi_max * (-4 * chi_max + tau0))) * SINH(3 * chi_max, STL) + 84 * COSH(3 * tau0, STL) * (40 * POW2(chi_max, EXACT) + 15 * POW2(POW2(chi_max, EXACT), EXACT) + 9 * POW2(POW3(chi_max, EXACT), EXACT)) - 16 * chi_max * (5 + 60 * POW2(chi_max, EXACT) + 108 * POW2(POW2(chi_max, EXACT), EXACT)) * tau0 + 10 * (8 + 18 * POW2(chi_max, EXACT) + 189 * POW2(POW2(chi_max, EXACT), EXACT)) * POW2(tau0, EXACT) - 120 * chi_max * (1 + 6 * POW2(chi_max, EXACT)) * POW3(tau0, EXACT) + 60 * POW2(POW2(tau0, EXACT), EXACT) + 9 * POW2(POW3(tau0, EXACT), EXACT) - 10 * POW3(chi_max, EXACT) * (15 * chi_max * COSH(3 * chi_max, STL) + 8 * SINH(3 * chi_max, STL))) + 14 * COSH(6 * tau0, STL) * (270 * POW2(POW2(chi_max, EXACT), EXACT) - 40 * (chi_max + 3 * POW3(chi_max, EXACT)) * SINH(3 * chi_max, STL) + (chi_max * (2855 + 3165 * POW2(chi_max, EXACT) + 162 * POW2(POW2(chi_max, EXACT), EXACT)) - 5 * (571 + 2016 * POW2(chi_max, EXACT) + 486 * POW2(POW2(chi_max, EXACT), EXACT)) * tau0 + 720 * chi_max * (14 + 9 * POW2(chi_max, EXACT)) * POW2(tau0, EXACT) - 60 * (56 + 117 * POW2(chi_max, EXACT)) * POW3(tau0, EXACT) + 3510 * chi_max * POW2(POW2(tau0, EXACT), EXACT) - 702 * POW2(tau0, EXACT) * POW3(tau0, EXACT)) * 
//	SINH(6 * chi_max, STL)) + 4 * (1260 * POW2(chi_max, EXACT) * tau0 * (2 + 3 * POW2(tau0, EXACT)) - 1890 * POW2(POW2(chi_max, EXACT), EXACT) * tau0 * (14 + 9 * POW2(tau0, EXACT)) + 756 * POW2(chi_max, EXACT) * POW3(chi_max, EXACT) * (23 + 27 * POW2(tau0, EXACT)) - tau0 * (1120 + 1680 * POW2(tau0, EXACT) + 756 * POW2(POW2(tau0, EXACT), EXACT) + 81 * POW2(POW3(tau0, EXACT), EXACT)) + 210 * POW3(chi_max, EXACT) * (32 + 27 * POW2(tau0, EXACT) * (2 + POW2(tau0, EXACT))) + 210 * POW3(chi_max, EXACT) * (8 * COSH(3 * chi_max, STL) + 15 * chi_max * SINH(3 * chi_max, STL))) * SINH(3 * tau0, STL) - 14 * (12 * POW3(chi_max, EXACT) * (10 + 27 * POW2(chi_max, EXACT)) - 40 * (chi_max + 3 * POW3(chi_max, EXACT)) * COSH(3 * chi_max, STL) + (chi_max * (2855 + 3165 * POW2(chi_max, EXACT) + 162 * POW2(POW2(chi_max, EXACT), EXACT) - 5 * (571 + 2016 * POW2(chi_max, EXACT) + 486 * POW2(POW2(chi_max, EXACT), EXACT)) * tau0 + 720 * chi_max * (14 + 9 * POW2(chi_max, EXACT)) * POW2(tau0, EXACT) - 60 * (56 + 117 * POW2(chi_max, EXACT)) * POW3(tau0, EXACT) + 3510 * chi_max * POW2(POW2(tau0, EXACT), EXACT) - 702 * POW2(tau0, EXACT) * POW3(tau0, EXACT)) * COSH(6 * chi_max, STL)) * SINH(6 * tau0, STL))))) / (204120 * POW3(chi_max, EXACT) * (-3 * tau0 + SINH(3 * tau0, STL)));
}

//Case 3: tau0 < chi_max
inline double averageDegreeFLRW_C3(const double &a, const double &chi_max, const double &tau0, const double &alpha, const double &delta)
{
	return 0.0;
}

//Wrapper Function
inline double averageDegreeFLRW(const double &a, const double &chi_max, const double &tau0, const double &alpha, const double &delta)
{
	if (tau0 > 2.0 * chi_max)
		return averageDegreeFLRW_C1(a, chi_max, tau0, alpha, delta);
	else if (chi_max < tau0 && tau0 < 2.0 * chi_max)
		return averageDegreeFLRW_C2(a, chi_max, tau0, alpha, delta);
	else if (tau0 < chi_max)
		return averageDegreeFLRW_C3(a, chi_max, tau0, alpha, delta);

	//This statement will never be called in practice
	return 0.0;
}*/

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
//Used to calculate geodesic distances in (embedded) universe
//For use with GNU Scientific Library
inline double embeddedZ1(double x, void *params)
{
	if (DEBUG)
		assert (params != NULL);

	double *p = (double*)params;

	double a = p[0];
	double alpha = p[1];
	double alpha2 = POW2(alpha, EXACT);

	if (DEBUG) {
		assert (a > 0.0);
		assert (alpha > 0.0);
	}

	return SQRT((1.0 / alpha2) + (POW2(a, EXACT) * x) / (alpha2 * alpha + POW3(x, EXACT)), STL);
}

inline double geodesicMaxRescaledTime(const double &lambda, const double &a, const bool &universe)
{
	if (DEBUG) {
		assert (lambda != 0.0);
		assert (a > 0.0);
	}

	if (lambda >= 0.0)
		return 0.0f;

	if (universe)
		return (2.0 / 3.0) * ASINH(POW(ABS(lambda, STL), -0.75, STL), STL, VERY_HIGH_PRECISION);
	else
		return ACOSH(POW(ABS(lambda, STL), -0.5, STL) / a, STL, VERY_HIGH_PRECISION);
}

//Integrands in Exact Geodesic Calculations
//For use with GNU Scientific Library

inline double deSitterDistKernel(double x, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (x >= 0);
	}

	double *p = (double*)params;
	double lambda = p[0];
	double a = p[1];

	if (DEBUG) {
		assert (lambda != 0.0);
		assert (a > 0.0);
	}

	return POW(ABS(1.0 / (lambda * POW2(a, EXACT) * POW2(COSH(x / a, STL), EXACT)) - 1.0, STL), -0.5, STL);
}

inline double flrwDistKernel(double x, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (x >= 0);
	}

	double *p = (double*)params;
	double lambda = p[0];
	double a = p[1];

	if (DEBUG) {
		assert (lambda != 0.0);
		assert (a > 0.0);
	}

	return POW(ABS(POW(SINH(1.5 * x / a, STL), -4.0 / 3.0, STL) / lambda - 1.0, STL), -0.5, STL);
}

//Returns the exact distance between two nodes
//O(xxx) Efficiency (revise this)
inline double distance(const double * const table, const float4 &node_a, const float &tau_a, const float4 &node_b, const float &tau_b, const int &dim, const Manifold &manifold, const double &a, const double &alpha, const long &size, const bool &universe, const bool &compact)
{
	if (DEBUG) {
		if (universe) {
			assert (table != NULL);
			assert (alpha > 0.0);
			assert (size > 0);
			assert (!compact);
		}
		assert (dim == 3);
		assert (manifold == DE_SITTER);
		assert (a > 0.0);
	}

	//Check if they are the same node
	if (node_a.w == node_b.w && node_a.x == node_b.x && node_a.y == node_b.y && node_a.z == node_b.z)
		return 0.0;

	FastIntMethod method;
	double (*kernel)(double x, void *params);
	double distance;
	double lambda;

	if (universe) {
		lambda = lookupValue4D(table, size, tau_a, tau_b, (alpha / a) * SQRT(flatProduct_v2(node_a, node_b), STL));
		kernel = &flrwDistKernel;
		method = QNG;
	} else {
		//float p2[3];
		//p2[0] = tau_a;
		//p2[1] = tau_b;
		//p2[2] = ACOS(sphProduct_v2(node_a, node_b), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);

		double x = 0.5;
		//if (!newton(&solveLambda4D, &x, 10000, TOL, &a, p2, NULL))
		//	return -1.0;
		//if (!bisection(&solveLambda4DBisec, &x, 10000, -100.0, 10000.0, TOL, false, &a, p2, NULL))
		//	return -1.0;
		lambda = x;
		kernel = &deSitterDistKernel;
		method = QAGS;
	}

	//DEBUG
	printf("Lambda: %f\n", lambda);
	//printf("Lambda: %f\tt1: %f\tt2: %f\n", lambda, tau_a, tau_b);

	//Check for NaN
	if (lambda != lambda)
		return -1.0;

	IntData idata = IntData();
	idata.limit = 50;
	idata.tol = 1e-5;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	
	double p[2];
	p[0] = lambda;
	p[1] = a;

	if (DEBUG)
		assert (lambda != 0.0);

	if (lambda > 0.0) {
		idata.lower = a * tau_a;
		idata.upper = a * tau_b;
		distance = integrate1D(kernel, (void*)p, &idata, method);
	} else {
		double tau_max = geodesicMaxRescaledTime(lambda, a, universe);

		idata.lower = a * tau_a;
		idata.upper = a * tau_max;
		distance = integrate1D(kernel, (void*)p, &idata, method);

		idata.lower = a * tau_b;
		distance += integrate1D(kernel, (void*)p, &idata, method);
	}

	gsl_integration_workspace_free(idata.workspace);

	return distance;
}

//Returns the embedded FLRW distance between two nodes
//O(xxx) Efficiency (revise this)
inline double distanceEmb(const float4 &node_a, const float &tau_a, const float4 &node_b, const float &tau_b, const int &dim, const Manifold &manifold, const double &a, const double &alpha, const bool &universe, const bool &compact)
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
		z1_a = POW(SINH(1.5 * tau_a, APPROX ? FAST : STL), power, APPROX ? FAST : STL);
		z1_b = POW(SINH(1.5 * tau_b, APPROX ? FAST : STL), power, APPROX ? FAST : STL);

		//Use Numerical Integration for z0
		idata.upper = alpha * z1_a;
		z0_a = integrate1D(&embeddedZ1, (void*)p, &idata, QNG);
		idata.upper = alpha * z1_b;
		z0_b = integrate1D(&embeddedZ1, (void*)p, &idata, QNG);
	} else {
		z0_a = SINH(tau_a, APPROX ? FAST : STL);
		z0_b = SINH(tau_b, APPROX ? FAST : STL);

		z1_a = COSH(tau_a, APPROX ? FAST : STL);
		z1_b = COSH(tau_b, APPROX ? FAST : STL);
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
