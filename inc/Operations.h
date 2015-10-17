#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include "Causet.h"
#include "Subroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//======================//
// Root-Finding Kernels //
//======================//

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

	return _tanx * (12.0 * (x * _tanx + _lncscx) + (6.0 * _lncscx - 5.0) * _cscx2 - 7.0) / (3.0 * HALF_PI * POW2(2.0 + _cscx2, EXACT)) - (k_tar / N_tar);
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

inline double tau0Flat(const double &x, const int &N_tar, const double &alpha, const double &delta, const double &a, const double &r_max)
{
	return SINH(3.0 * x, APPROX ? FAST : STL) - 3.0 * (x + static_cast<double>(N_tar) / (1.5 * M_PI * delta * a * POW3(alpha * r_max, EXACT)));
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
	#if DEBUG
	assert (p2 != NULL);
	assert (p3 != NULL);
	assert (p2[0] > 0.0f);			//k_tar
	assert (p3[0] > 0);			//N_tar
	assert (p3[1] == 1 || p3[1] == 3);	//dim
	#endif

	return ((p3[1] == 1) ?
		-1.0 * eta02D(x, p3[0], p2[0]) / eta0Prime2D(x) :
		-1.0 * zeta4D(x, p3[0], p2[0]) / zetaPrime4D(x));
}

//Returns zeta Residual in Bisection Algorithm
//Used in 1+1 and 3+1 Causets
inline double solveZetaBisec(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p2 != NULL);
	assert (p3 != NULL);
	assert (p2[0] > 0.0f);			//k_tar
	assert (p3[0] > 0);			//N_tar
	assert (p3[1] == 1 || p3[1] == 3);	//dim
	#endif

	return ((p3[1] == 1) ?
		eta02D(x, p3[0], p2[0]) :
		zeta4D(x, p3[0], p2[0]));
}

//Returns tau0 Residual
//Used in Universe Causet
inline double solveTau0Compact(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p3 != NULL);
	assert (p1[0] > 0.0);	//alpha
	assert (p1[1] > 0.0);	//delta
	assert (p1[2] > 0.0);	//a
	assert (p3[0] > 0);	//N_tar
	#endif

	return (-1.0 * tau0Compact(x, p3[0], p1[0], p1[1], p1[2]) / tau0Prime(x));
}

inline double solveTau0Flat(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p3 != NULL);
	assert (p1[0] > 0.0);	//alpha
	assert (p1[1] > 0.0);	//delta
	assert (p1[2] > 0.0);	//a
	assert (p1[3] > 0.0);	//r_max
	assert (p3[0] > 0);	//N_tar
	#endif

	return (-1.0 * tau0Flat(x, p3[0], p1[0], p1[1], p1[2], p1[3]) / tau0Prime(x));
}

//Returns tau Residual
//Used in 3+1 Causet
inline double solveTau(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0 && p1[0] < HALF_PI);	//zeta
	assert (p1[1] > 0.0 && p1[1] < 1.0);		//rval
	#endif

	return (-1.0 * tau4D(x, p1[0], p1[1]) / tauPrime4D(x, p1[0]));
}

//Returns tau Residual
//Used in Universe Causet
inline double solveTauUniverse(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);			//tau0
	assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	#endif

	return (-1.0 * tauUniverse(x, p1[0], p1[1]) / tauPrimeUniverse(x, p1[0]));
}

//Returns tau Residual in Bisection Algorithm
//Used in Universe Causet
inline double solveTauUnivBisec(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);			//tau0
	assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	#endif

	return tauUniverse(x, p1[0], p1[1]);
}

//Returns theta1 Residual
//Used in 3+1 and Universe Causets
inline double solveTheta1(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0 && p1[0] < 1.0);	//rval
	#endif

	return (-1.0 * theta1_4D(x, p1[0]) / theta1_Prime4D(x));
}

//======================//
// Constrain Causal Set //
//======================//
//
//See NetworkCreator.cu/initVars() for details

inline double solveDeltaCompact(const int &N_tar, const double &a, const double &tau0, const double &alpha)
{
	#if DEBUG
	assert (N_tar > 0);
	assert (a > 0.0);
	assert (tau0 > 0.0);
	assert (alpha > 0.0);
	#endif

	double delta;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		delta = exp(LOG(6.0 * N_tar / (POW2(M_PI, EXACT) * a * POW3(alpha, EXACT)), STL) - 3.0 * tau0);
	else
		delta = 3.0 * N_tar / (POW2(M_PI, EXACT) * a * POW3(alpha, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));

	#if DEBUG
	assert (delta > 0.0);
	#endif

	return delta;
}

inline double solveDeltaFlat(const int &N_tar, const double &a, const double &r_max, const double &tau0, const double &alpha)
{
	#if DEBUG
	assert (N_tar > 0);
	assert (a > 0.0);
	assert (r_max > 0.0);
	assert (tau0 > 0.0);
	assert (alpha > 0.0);
	#endif

	double delta;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		delta = exp(LOG(9.0 * N_tar / (M_PI * a * POW3(alpha * r_max, EXACT)), STL) - 3.0 * tau0);
	else
		delta = 9.0 * N_tar / (2.0 * M_PI * a * POW3(alpha * r_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0));

	#if DEBUG
	assert (delta > 0.0);
	#endif

	return delta;
}

inline int solveNtarCompact(const double &a, const double &tau0, const double &alpha, const double &delta)
{
	#if DEBUG
	assert (a > 0.0);
	assert (tau0 > 0.0);
	assert (alpha > 0.0);
	assert (delta > 0.0);
	#endif

	int N_tar;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		N_tar = static_cast<int>(exp(LOG(6.0 / (POW2(M_PI, EXACT) * delta * a * POW3(alpha, EXACT)), EXACT) - 3.0 * tau0));
	else
		N_tar = static_cast<int>(POW2(M_PI, EXACT) * delta * a * POW3(alpha, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0) / 3.0);

	#if DEBUG
	assert (N_tar > 0);
	#endif

	return N_tar;
}

inline int solveNtarFlat(const double &a, const double &r_max, const double &tau0, const double &alpha, const double &delta)
{
	#if DEBUG
	assert (a > 0.0);
	assert (r_max > 0.0);
	assert (tau0 > 0.0);
	assert (alpha > 0.0);
	assert (delta > 0.0);
	#endif

	int N_tar;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		N_tar = static_cast<int>(exp(LOG(9.0 / (M_PI * delta * a * POW3(alpha * r_max, EXACT)), EXACT) - 3.0 * tau0));
	else
		N_tar = static_cast<int>(2.0 * M_PI * delta * a * POW3(alpha * r_max, EXACT) * (SINH(3.0 * tau0, STL) - 3.0 * tau0) / 9.0);

	#if DEBUG
	assert (N_tar > 0);
	#endif

	return N_tar;
}

inline double solveAlphaCompact(const int &N_tar, const double &a, const double &tau0, const double &delta)
{
	#if DEBUG
	assert (N_tar > 0);
	assert (a > 0.0);
	assert (tau0 > 0.0);
	assert (delta > 0.0);
	#endif

	double alpha;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		alpha = exp(LOG(6.0 * N_tar / (POW2(M_PI, EXACT) * delta * a), STL) / 3.0 - tau0);
	else
		alpha = POW(3.0 * N_tar / (POW2(M_PI, EXACT) * delta * a * (SINH(3.0 * tau0, STL) - 3.0 * tau0)), (1.0 / 3.0), STL);

	#if DEBUG
	assert (alpha > 0.0);
	#endif

	return alpha;
}

inline double solveAlphaFlat(const int &N_tar, const double &a, const double &r_max, const double &tau0, const double &delta)
{
	#if DEBUG
	assert (N_tar > 0);
	assert (a > 0.0);
	assert (r_max > 0.0);
	assert (tau0 > 0.0);
	assert (delta > 0.0);
	#endif

	double alpha;
	if (tau0 > LOG(MTAU, STL) / 3.0)
		alpha = exp(LOG(9.0 * N_tar / (M_PI * delta * a), STL) / 3.0 - tau0);
	else
		alpha = POW(9.0 * N_tar / (2.0 * M_PI * delta * a * (SINH(3.0 * tau0, STL) - 3.0 * tau0)), (1.0 / 3.0), STL);

	#if DEBUG
	assert (alpha > 0.0);
	#endif

	return alpha;
}

//=========================//
// Spatial Length Formulae //
//=========================//

//X1 Coordinate of de Sitter 3-Metric
inline float X1_SPH(const float &theta1)
{
	return static_cast<float>(COS(theta1, APPROX ? FAST : STL));
}

//X2 Coordinate of Spherical 3-Metric
inline float X2_SPH(const float &theta1, const float &theta2)
{
	return static_cast<float>(SIN(theta1, APPROX ? FAST : STL) * COS(theta2, APPROX ? FAST : STL));
}

//X3 Coordinate of Spherical 3-Metric
inline float X3_SPH(const float &theta1, const float &theta2, const float &theta3)
{
	return static_cast<float>(SIN(theta1, APPROX ? FAST : STL) * SIN(theta2, APPROX ? FAST : STL) * COS(theta3, APPROX ? FAST : STL));
}

//X4 Coordinate of Spherical 3-Metric
inline float X4_SPH(const float &theta1, const float &theta2, const float &theta3)
{
	return static_cast<float>(SIN(theta1, APPROX ? FAST : STL) * SIN(theta2, APPROX ? FAST : STL) * SIN(theta3, APPROX ? FAST : STL));
}

//X Coordinate from Flat 3-Metric
inline float X_FLAT(const float &theta1, const float &theta2, const float &theta3)
{
	return static_cast<float>(theta1 * SIN(theta2, APPROX ? FAST : STL) * COS(theta3, APPROX ? FAST : STL));
}

//Y Coordinate from Flat 3-Metric
inline float Y_FLAT(const float &theta1, const float &theta2, const float &theta3)
{
	return static_cast<float>(theta1 * SIN(theta2, APPROX ? FAST : STL) * SIN(theta3, APPROX ? FAST : STL));
}

//Z Coordinate from Flat 3-Metric
inline float Z_FLAT(const float &theta1, const float &theta2)
{
	return static_cast<float>(theta1 * COS(theta2, APPROX ? FAST : STL));
}

//Spherical Inner Product
//Returns COS(angle) between two points on unit sphere
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

//=========================//
// Node Relation Algorithm //
//=========================//

//Assumes coordinates have been temporally ordered
inline bool nodesAreRelated(Coordinates *c, const int &N_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const bool &compact, int past_idx, int future_idx, double *omega12)
{
	#if DEBUG
	assert (!c->isNull());
	assert (dim == 1 || dim == 3);
	assert (manifold == DE_SITTER || manifold == DUST || manifold == FLRW);

	if (dim == 1)
		assert (c->getDim() == 2);
	else if (dim == 3) {
		assert (c->getDim() == 4);
		assert (c->w() != NULL);
		assert (c->z() != NULL);
	}

	assert (c->x() != NULL);
	assert (c->y() != NULL);

	assert (N_tar > 0);
	assert (a > 0.0);
	if (manifold == DE_SITTER) {
		if (compact) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	} else if (manifold == FLRW) {
		assert (zeta < HALF_PI);
		assert (alpha > 0.0);
	}
	if (!compact)
		assert (r_max > 0.0);
		
	assert (past_idx >= 0 && past_idx < N_tar);
	assert (future_idx >= 0 && future_idx < N_tar);
	assert (past_idx < future_idx);
	#endif

	float dt = 0.0f, dx = 0.0f;

	//Temporal Interval
	if (dim == 1)
		dt = c->x(future_idx) - c->x(past_idx);
	else if (dim == 3)
		dt = c->w(future_idx) - c->w(past_idx);

	#if DEBUG
	assert (dt >= 0.0f);
	if (!compact && manifold == DE_SITTER)
		assert (dt <= zeta - zeta1);
	else
		assert (dt <= static_cast<float>(HALF_PI - zeta));
	#endif

	//Spatial Interval
	if (dim == 1)
		dx = static_cast<float>(M_PI - ABS(M_PI - ABS(static_cast<double>(c->y(future_idx) - c->y(past_idx)), STL), STL));
	else if (dim == 3) {
		if (compact) {
			//Spherical Law of Cosines
			#if DIST_V2
				dx = static_cast<float>(ACOS(static_cast<double>(sphProduct_v2(c->getFloat4(past_idx), c->getFloat4(future_idx))), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			#else
				dx = static_cast<float>(ACOS(static_cast<double>(sphProduct_v1(c->getFloat4(past_idx), c->getFloat4(future_idx))), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			#endif
		} else {
			//Law of Cosines
			#if DIST_V2
				dx = static_cast<float>(SQRT(static_cast<double>(flatProduct_v2(c->getFloat4(past_idx), c->getFloat4(future_idx))), APPROX ? BITWISE : STL));
			#else
				dx = static_cast<float>(SQRT(static_cast<double>(flatProduct_v1(c->getFloat4(past_idx), c->getFloat4(future_idx))), APPROX ? BITWISE : STL));
			#endif
		}
	}

	#if DEBUG
	if (compact)
		assert (dx >= 0.0f && dx <= static_cast<float>(M_PI));
	else
		assert (dx >= 0.0f && dx <= 2.0f * static_cast<float>(r_max));
	#endif

	if (omega12 != NULL)
		*omega12 = dx;

	if (dx < dt)
		return true;
	else
		return false;
}

//=================================//
// Conformal/Cosmic Time Relations //
//=================================//

//Formulae in de Sitter

//Conformal to Rescaled Time (de Sitter compact)
inline double etaToTauCompact(const double eta)
{
	#if DEBUG
	assert (eta >= 0.0 && eta < HALF_PI);
	#endif

	return ACOSH(1.0 / COS(eta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Conformal to Rescaled Time (de Sitter flat)
inline double etaToTauFlat(const double eta)
{
	#if DEBUG
	assert (eta < 0.0 && -1.0 * eta < HALF_PI);
	#endif

	return -1.0 * LOG(-1.0 * eta, APPROX ? FAST : STL);
}

//Rescaled to Conformal Time (de Sitter compact)
inline double tauToEtaCompact(const double tau)
{
	#if DEBUG
	assert (tau > 0.0);
	#endif

	return ACOS(1.0 / COSH(tau, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Rescaled to Conformal Time (de Sitter flat)
inline double tauToEtaFlat(const double tau)
{
	#if DEBUG
	assert (tau > 0.0);
	#endif

	return -exp(-tau);
}

//Formulae for Dust

//Conformal to Rescaled Time (Dust)
inline double etaToTauDust(const double eta, const double a, const double alpha)
{
	#if DEBUG
	assert (eta > 0.0);
	assert (a > 0.0);
	assert (alpha > 0.0);
	#endif

	return POW3(eta * alpha / a, EXACT) / 12.0;
}

//Rescaled to Conformal Time (Dust)
inline double tauToEtaDust(const double tau, const double a, const double alpha)
{
	#if DEBUG
	assert (tau > 0.0);
	assert (a > 0.0);
	assert (alpha > 0.0);
	#endif

	return POW(12.0 * tau, 1.0 / 3.0, STL) * a / alpha;
}

//Formulae in FLRW

//For use with GNU Scientific Library
inline double tauToEtaFLRW(double tau, void *params)
{
	#if DEBUG
	assert (tau > 0.0);
	#endif

	return POW(SINH(1.5 * tau, APPROX ? FAST : STL), (-2.0 / 3.0), APPROX ? FAST : STL);
}

//'Exact' Solution (Hypergeomtric Series)
inline double tauToEtaFLRWExact(const double &tau, const double a, const double alpha)
{
	#if DEBUG
	assert (tau > 0.0);
	assert (a > 0.0);
	assert (alpha > 0.0);
	#endif

	double eta = 0.0;

	//Used for _2F1
	double f;
	double err = 1.0E-10;
	int nterms = -1;

	//Determine which transformation of 2F1 is used
	double z = 1.0 / POW2(COSH(1.5 * tau, APPROX ? FAST : STL), EXACT);
	double w;
	if (z >= 0.0 && z <= 0.5) {
		w = z;
		_2F1(1.0 / 3.0, 5.0 / 6.0, 4.0 / 3.0, w, &f, &err, &nterms, false);
		eta = SQRT(3.0 * POW3(M_PI, EXACT), STL) / (GAMMA(5.0 / 6.0, STL) * GAMMA(-4.0 / 3.0, STL)) - POW(w, 1.0 / 3.0, STL) * f;
	} else if (z > 0.5 && z <= 1.0) {
		w = 1 - z;
		_2F1(0.5, 1.0, 7.0 / 6.0, w, &f, &err, &nterms, false);
		eta = 2.0 * POW(z * SQRT(w, STL), 1.0 / 3.0, STL) * f;
	} else
		//This should never be reached
		return NAN;
	//printf("tau: %.16e\teta: %.16e\n", tau, eta);

	eta *= a / alpha;

	#if DEBUG
	assert (eta == eta);
	assert (eta >= 0.0);
	#endif

	return eta;
}

//Gives Input to 'ctuc' Lookup Table
inline double etaToTauFLRW(const double &eta, const double &a, const double &alpha)
{
	#if DEBUG
	assert (eta > 0.0);
	assert (a > 0.0);
	assert (alpha > 0.0);
	#endif

	double g = 9.0 * GAMMA(2.0 / 3.0, STL) * alpha * eta / a;
	g -= 4.0 * SQRT(3.0, APPROX ? BITWISE : STL) * POW(M_PI, 1.5, STL) / GAMMA(5.0 / 6.0, STL);
	g /= 3.0 * GAMMA(-1.0 / 3.0, STL);

	#if DEBUG
	assert (g > 0.0);
	#endif

	return g;
}

//=========================//
// Average Degree Formulae //
//=========================//

//Rescaled Average Degree in Flat (K = 0) de Sitter Causet

//This kernel is used in numerical integration
//This result should be multipled by 4pi*(eta0*eta1)^3/(eta1^3 - eta0^3)
inline double rescaledDegreeDeSitterFlat(int dim, double x[], double *params)
{
	#if DEBUG
	assert (dim > 0);
	assert (x[0] < 0.0 && x[0] > -1.0 * HALF_PI);
	assert (x[1] < 0.0 && x[1] > -1.0 * HALF_PI);
	#endif

	//Identify x[0] with eta' coordinate
	//Identify x[1] with eta'' coordinate
	
	double t = POW2(POW2(x[0] * x[1], EXACT), EXACT);
	return ABS(POW3(x[0] - x[1], EXACT), STL) / t;
}

//Rescaled Average Degree in Dusty Causet
//This result should be multiplied by (108pi / tau0^3)
inline double rescaledDegreeDust(int dim, double x[], double *params)
{
	#if DEBUG
	assert (dim > 0);
	assert (x[0] > 0.0);
	assert (x[1] > 0.0);
	#endif

	//Identify x[0] with the tau' coordinate
	//Identify x[1] with the tau'' coordinate
	
	double h1 = x[0] * x[1];
	double h2 = ABS(POW(x[0], 1.0 / 3.0, STL) - POW(x[1], 1.0 / 3.0, STL), STL);

	return POW2(h1, EXACT) * POW3(h2, EXACT);
}

//Rescaled Average Degree in Non-Compact FLRW Causet

//This is a kernel used in numerical integration
//Note to get the (non-compact) rescaled average degree this result must still be
//multiplied by 8pi / (sinh(3tau0)-3tau0)
inline double rescaledDegreeFLRW_NC(int dim, double x[], double *params)
{
	#if DEBUG
	assert (dim > 0);
	assert (x[0] > 0.0);
	assert (x[1] > 0.0);
	#endif

	//Identify x[0] with tau' coordinate
	//Identify x[1] with tau'' coordinate

	double h1 = tauToEtaFLRWExact(x[0], 1.0, 1.0);
	double h2 = tauToEtaFLRWExact(x[1], 1.0, 1.0);

	double s1 = POW2(SINH(1.5 * x[0], APPROX ? FAST : STL), EXACT);
	double s2 = POW2(SINH(1.5 * x[1], APPROX ? FAST : STL), EXACT);

	return s1 * s2 * ABS(POW3(h2 - h1, EXACT), STL);
}

//Rescaled Average Degree in Compact FLRW Causet

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
		double z = -1.0 * POW3(r, EXACT);
		_2F1(1.0 / 6.0, 0.5, 7.0 / 6.0, z, &f, &err, &nterms, false);
		_xi = 2.0 * SQRT(r, STL) * f;
	} else {
		double z = -1.0 / POW3(r, EXACT);
		_2F1(1.0 / 3.0, 0.5, 4.0 / 3.0, z, &f, &err, &nterms, false);
		_xi = SQRT(4.0 / M_PI, STL) * GAMMA(7.0 / 6.0, STL) * GAMMA(1.0 / 3.0, STL) - f / r;
	}

	return _xi;
}

//This is a kernel used in numerical integration
//Note to get the (compact) rescaled averge degree this result must still be
//multiplied by 8pi/(sinh(3tau0)-3tau0)
inline double rescaledDegreeFLRW(int dim, double x[], double *params)
{
	#if DEBUG
	assert (dim > 0);
	assert (x[0] > 0.0);
	assert (x[1] > 0.0);
	#endif

	//Identify x[0] with x coordinate
	//Identify x[1] with r coordinate

	double z;

	z = POW3(ABS(xi(x[0]) - xi(x[1]), STL), EXACT) * POW2(x[0], EXACT) * POW3(x[1], EXACT) * SQRT(x[1], STL);
	z /= (SQRT(1.0 + 1.0 / POW3(x[0], EXACT), STL) * SQRT(1.0 + POW3(x[1], EXACT), STL));

	#if DEBUG
	assert (z > 0.0);
	#endif

	return z;
}

//Average Degree in Compact FLRW Causet (not rescaled)

//Gives rescaled scale factor as a function of eta
//Uses 'ctuc' lookup table
inline double rescaledScaleFactor(double *table, double size, double eta, double a, double alpha)
{
	#if DEBUG
	assert (table != NULL);
	assert (size > 0.0);
	assert (eta > 0.0);
	assert (a > 0.0);
	assert (alpha > 0.0);
	#endif

	double g = etaToTauFLRW(eta, a, alpha);
	#if DEBUG
	assert (g > 0.0);
	#endif

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
	
	#if DEBUG
	assert (tau > 0.0);
	#endif
		
	return POW(SINH(1.5 * tau, APPROX ? FAST : STL), 2.0 / 3.0, APPROX ? FAST : STL);
}

//This is a kernel used in numerical integration
//Note to get the average degree this result must still be
//multipled by (4pi/3)*delta*alpha^4/psi
inline double averageDegreeFLRW(int dim, double x[], double *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (dim > 0);
	assert (x[0] > 0.0);
	assert (x[1] > 0.0);
	assert (params[0] > 0.0);
	assert (params[1] > 0.0);
	assert (params[2] > 0.0);
	#endif

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

	#if DEBUG
	assert (z > 0.0);
	#endif

	return z;
}

//For use with GNU Scientific Library
inline double psi(double eta, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (eta > 0.0);
	#endif

	//Identify params[0] with a
	//Identify params[1] with alpha
	//Identify params[2] with size
	//Identify params[3] with table
	
	return POW2(POW2(rescaledScaleFactor(&((double*)params)[3], ((double*)params)[2], eta, ((double*)params)[0], ((double*)params)[1]), EXACT), EXACT);
}

//=======================//
// Degree Field Formulae //
//=======================//

//For use with GNU Scientific Library
inline double degreeFieldTheory(double eta, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (eta > 0.0);
	#endif

	//Identify params[0] with eta_m
	//Identify params[1] with a
	//Identify params[2] with alpha
	//Identify params[3] with size
	//Identify params[4] with table
	
	return POW3(ABS(((double*)params)[0] - eta, STL), EXACT) * POW2(POW2(rescaledScaleFactor(&((double*)params)[4], ((double*)params)[3], eta, ((double*)params)[1], ((double*)params)[2]), EXACT), EXACT);
}

#endif
