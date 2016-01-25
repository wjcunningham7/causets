#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include "Causet.h"
#include "Subroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

//======================//
// Root-Finding Kernels //
//======================//

inline double eta4Dfunc(const double &x, const double &a)
{
	return -a + x * (6.0 + x * (3.0 * a + x * (-4.0 + x * (-3.0 * a + x * (6.0 + a * x)))));
}

inline double eta4DfuncPrime(const double &x, const double &a)
{
	return 6.0 * (1.0 + x * (a + x * (-2.0 + x * (-2.0 * a + x * (5.0 + a * x)))));
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

//Returns eta Residual
//Used in 3+1 Spherical de Sitter Causet
inline double solveEtaFunc(const double &x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);	//"a" (not same as pseudoradius)
	#endif

	return (-1.0 * eta4Dfunc(x, p1[0]) / eta4DfuncPrime(x, p1[0]));
}

//Returns tau Residual
//Used in 3+1 Spherical de Sitter Causet
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
//Used in FLRW Causet
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
//Used in FLRW Causet
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

//Embedded form
inline float sphEmbProduct(const float5 &sc0, const float5 &sc1)
{
	return sc0.w * sc1.w + sc0.x * sc1.x + sc0.y * sc1.y + sc0.z * sc1.z; 
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

//Embedded form
inline float flatEmbProduct(const float5 &sc0, const float5 &sc1)
{
	return POW2(sc0.x - sc1.x, EXACT) + POW2(sc0.y - sc1.y, EXACT) + POW2(sc0.z - sc1.z, EXACT);
}

//=========================//
// Node Relation Algorithm //
//=========================//

//Assumes coordinates have been temporally ordered
inline bool nodesAreRelated(Coordinates *c, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, int past_idx, int future_idx, double *omega12)
{
	#if DEBUG
	assert (!c->isNull());
	assert (get_stdim(spacetime) & (2 | 4));
	assert (get_manifold(spacetime) & (DE_SITTER | DUST | FLRW));

	if (get_stdim(spacetime) == 2) {
		#if EMBED_NODES
		assert (c->getDim() == 3);
		assert (c->z() != NULL);
		#else
		assert (c->getDim() == 2);
		#endif
	} else if (get_stdim(spacetime) == 4) {
		#if EMBED_NODES
		assert (c->getDim() == 5);
		assert (c->v() != NULL);
		#else
		assert (c->getDim() == 4);
		#endif
		assert (c->w() != NULL);
		assert (c->z() != NULL);
	}

	assert (c->x() != NULL);
	assert (c->y() != NULL);

	assert (N_tar > 0);
	assert (a > 0.0);
	if (get_manifold(spacetime) & DE_SITTER) {
		if (get_curvature(spacetime) & POSITIVE) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else if (get_curvature(spacetime) & FLAT) {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	} else if (get_manifold(spacetime) & (DUST | FLRW)) {
		assert (zeta < HALF_PI);
		assert (alpha > 0.0);
	}
	if (get_curvature(spacetime) & FLAT)
		assert (r_max > 0.0);
		
	assert (past_idx >= 0 && past_idx < N_tar);
	assert (future_idx >= 0 && future_idx < N_tar);
	assert (past_idx != future_idx);
	#endif

	float dt = 0.0f, dx = 0.0f;

	if (future_idx < past_idx) {
		//Bitwise swap
		past_idx ^= future_idx;
		future_idx ^= past_idx;
		past_idx ^= future_idx;
	}

	//Temporal Interval
	if (get_stdim(spacetime) == 2)
		dt = c->x(future_idx) - c->x(past_idx);
	else if (get_stdim(spacetime) == 4)
		#if EMBED_NODES
		dt = c->v(future_idx) - c->v(past_idx);
		#else
		dt = c->w(future_idx) - c->w(past_idx);
		#endif

	#if DEBUG
	assert (dt >= 0.0f);
	if ((get_curvature(spacetime) & FLAT) && (get_manifold(spacetime) & DE_SITTER))
		assert (dt <= zeta - zeta1);
	else {
		if (get_symmetry(spacetime) & SYMMETRIC)
			assert (dt <= 2.0f * static_cast<float>(HALF_PI - zeta));
		else
			assert (dt <= static_cast<float>(HALF_PI - zeta));
	}
	#endif

	//Spatial Interval
	if (get_stdim(spacetime) == 2) {
		#if EMBED_NODES
		float phi1 = atan2(c->z(future_idx), c->y(future_idx));
		float phi2 = atan2(c->z(past_idx), c->y(past_idx));
		dx = M_PI - fabs(M_PI - fabs(phi2 - phi1));
		#else
		dx = static_cast<float>(M_PI - ABS(M_PI - ABS(static_cast<double>(c->y(future_idx) - c->y(past_idx)), STL), STL));
		#endif
	} else if (get_stdim(spacetime) == 4) {
		if (get_curvature(spacetime) & POSITIVE) {
			//Spherical Law of Cosines
			#if EMBED_NODES
			dx = acosf(sphEmbProduct(c->getFloat5(past_idx), c->getFloat5(future_idx)));
			#else
			#if DIST_V2
			dx = static_cast<float>(ACOS(static_cast<double>(sphProduct_v2(c->getFloat4(past_idx), c->getFloat4(future_idx))), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			#else
			dx = static_cast<float>(ACOS(static_cast<double>(sphProduct_v1(c->getFloat4(past_idx), c->getFloat4(future_idx))), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION));
			#endif
			#endif
		} else if (get_curvature(spacetime) & FLAT) {
			//Euclidean Law of Cosines
			#if EMBED_NODES
			//float fep = flatEmbProduct(c->getFloat5(past_idx), c->getFloat5(future_idx));
			//printf("embProduct: %f\n", fep);
			//if (fep < 0.0f)
			//	printChk();
			dx = sqrtf(flatEmbProduct(c->getFloat5(past_idx), c->getFloat5(future_idx)));
			#else
			#if DIST_V2
			dx = static_cast<float>(SQRT(static_cast<double>(flatProduct_v2(c->getFloat4(past_idx), c->getFloat4(future_idx))), APPROX ? BITWISE : STL));
			#else
			dx = static_cast<float>(SQRT(static_cast<double>(flatProduct_v1(c->getFloat4(past_idx), c->getFloat4(future_idx))), APPROX ? BITWISE : STL));
			#endif
			#endif
		}
	}

	//printf("dx: %f\n", dx);

	#if DEBUG
	if (get_curvature(spacetime) & POSITIVE)
		assert (dx >= 0.0f && dx <= static_cast<float>(M_PI));
	else if (get_curvature(spacetime) & FLAT)
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

//Conformal to Rescaled Time (de Sitter, spherical foliation)
inline double etaToTauSph(const double eta)
{
	#if DEBUG
	assert (fabs(eta) < HALF_PI);
	#endif

	return SGN(eta, DEF) * ACOSH(1.0 / COS(eta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Conformal to Rescaled Time (de Sitter flat foliation)
inline double etaToTauFlat(const double eta)
{
	#if DEBUG
	assert (eta > -1.0f && eta < 0.0);
	#endif

	return -LOG(-eta, APPROX ? FAST : STL);
}

//Rescaled to Conformal Time (de Sitter spherical foliation)
inline double tauToEtaSph(const double tau)
{
	return SGN(tau, DEF) * ACOS(1.0 / COSH(tau, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Rescaled to Conformal Time (de Sitter flat foliation)
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

//Average Degree in Closed (K = 1) Symmetric de Sitter Causet

//This kernel is used in numerical integration
//This gives T2/2 - only part of the expression - see notes
inline double averageDegreeSym(double eta, void *params)
{
	#if DEBUG
	assert (cos(eta) >= 0.0);
	#endif

	double t = 1.0 / cos(eta);
	return eta * eta * t * t * t * t;
}

//Rescaled Average Degree in Dusty Causet

//This kernel is used in numerical integration
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

//================//
// Action Formula //
//================//

//Calculate the action from the abundancy intervals
//The parameter 'lk' is taken to be expressed in units of the graph discreteness length 'l'
inline double calcAction(const int * const cardinalities, const int stdim, const double &lk, const bool &smeared)
{
	#if DEBUG
	assert (cardinalities != NULL);
	assert (stdim == 2 || stdim == 4);
	assert (lk > 0.0);
	#endif

	double action = 0.0;

	if (smeared) {
		double epsilon = POW(lk, -stdim, STL);
		double eps1 = epsilon / (1.0 - epsilon);
		double ni;
		int i;

		for (i = 0; i < cardinalities[0] - 3; i++) {
			ni = static_cast<double>(cardinalities[i+1]);
			if (stdim == 2)
				action += ni * POW(1.0 - epsilon, i, STL) * (1.0 - 2.0 * eps1 * i + 0.5 * POW2(eps1, EXACT) * i * (i - 1.0));
			else if (stdim == 4)
				action += ni * POW(1.0 - epsilon, i, STL) * (1.0 - 9.0 * eps1 * i + 8.0 * POW2(eps1, EXACT) * i * (i - 1.0) - (4.0 / 3.0) * POW3(eps1, EXACT) * i * (i - 1.0) * (i - 2.0));
			else
				action = NAN;
		}

		if (stdim == 2)
			action = 2.0 * epsilon * (cardinalities[0] - 2.0 * epsilon * action);
		else if (stdim == 4)
			action = (4.0 / sqrt(6.0)) * (sqrt(epsilon) * cardinalities[0] - POW(epsilon, 1.5, STL) * action);
		else
			action = NAN;
	} else {
		if (stdim == 2)
			action = 2.0 * (cardinalities[0] - 2.0 * (cardinalities[1] - 2.0 * cardinalities[2] + cardinalities[3]));
		else if (stdim == 4)
			action = (4.0 / sqrt(6.0)) * (cardinalities[0] - cardinalities[1] + 9.0 * cardinalities[2] - 16.0 * cardinalities[3] + 8.0 * cardinalities[4]);
		else
			action = NAN;
	}

	return action;
}

#endif
