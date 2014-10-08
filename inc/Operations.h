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
	double _tanx = TAN(static_cast<float>(x), APPROX ? FAST : STL);
	return ((x / _tanx - LOG(COS(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL) - 1.0) / (_tanx * HALF_PI)) - (k_tar / N_tar);
}

inline double eta0Prime2D(const double &x)
{
	double _cotx = 1.0 / TAN(static_cast<float>(x), APPROX ? FAST : STL);
	double _cscx2 = 1.0 / POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _lnsecx = -1.0 * LOG(COS(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL);

	return (_cotx * (_cotx - x * _cscx2) + 1 - _cscx2 * (_lnsecx + x * _cotx - 1)) / HALF_PI;
}

inline double zeta4D(const double &x, const int &N_tar, const float &k_tar)
{
	double _tanx = TAN(static_cast<float>(x), APPROX ? FAST : STL);
	double _cscx2 = 1.0 / POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _lncscx = 0.5 * LOG(_cscx2, APPROX ? FAST : STL);

	return _tanx * (12.0 * (x * _tanx + _lncscx) + (6.0 * _lncscx - 5.0) * _cscx2 - 7.0) / (3.0 * HALF_PI * POW2(2.0f + _cscx2, EXACT)) - (k_tar / N_tar);
}

inline double zetaPrime4D(const double &x)
{
	double _cscx2 = 1.0 / POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _sinx3 = POW3(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _sinx4 = POW(SIN(static_cast<float>(x), APPROX ? FAST : STL), 4.0, APPROX ? FAST : STL);
	double _cosx3 = POW3(COS(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _lncscx = -1.0 * LOG(SIN(static_cast<float>(x), APPROX ? FAST : STL), APPROX ? FAST : STL);

	return (3.0 * (COS(5.0f * x, APPROX ? FAST : STL) - 32.0 * (M_PI - 2.0 * x) * _sinx3) + COS(static_cast<float>(x), APPROX ? FAST : STL) * (84.0 - 72.0 * _lncscx) + COS(3.0 * static_cast<float>(x), APPROX ? FAST : STL) * (24.0 * _lncscx - 31.0)) / (-4.0 * M_PI * _sinx4 * _cosx3 * POW3((2.0f + _cscx2), EXACT));
}

inline double tau0(const double &x, const int &N_tar, const double &alpha, const double &delta, const double &a)
{
	return SINH(3.0f * x, APPROX ? FAST : STL) - 3.0 * (x + static_cast<double>(N_tar) / (POW2(static_cast<float>(M_PI), EXACT) * delta * a * POW3(static_cast<float>(alpha), EXACT)));
}

inline double tau0Prime(const double &x)
{
	return 3.0 * (COSH(3.0 * static_cast<float>(x), APPROX ? FAST : STL) - 1.0);
}

inline double tau4D(const double &x, const double &zeta, const double &rval)
{
	double _coshx2 = POW2(COSH(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _sinz2 = POW2(SIN(static_cast<float>(zeta), APPROX ? FAST : STL), EXACT);

	return (2.0 + _coshx2) * SINH(static_cast<float>(x), APPROX ? FAST : STL) * TAN(static_cast<float>(zeta), APPROX ? FAST : STL) * _sinz2 / (2.0 * _sinz2 + 1) - rval;
}

inline double tauPrime4D(const double &x, const double &zeta)
{
	double _coshx2 = POW2(COSH(static_cast<float>(x), APPROX ? FAST : STL), EXACT);
	double _coshx4 = POW2(static_cast<float>(_coshx2), EXACT);
	double _sinz2 = POW2(SIN(static_cast<float>(zeta), APPROX ? FAST : STL), EXACT);

	return 3.0 * _coshx4 * TAN(static_cast<float>(zeta), APPROX ? FAST : STL) * _sinz2 / (2.0 * _sinz2 + 1);
}

inline double tauUniverse(const double &x, const double &tau0, const double &rval)
{
	return (SINH(3.0f * x, APPROX ? FAST : STL) - 3.0 * x) / (SINH(3.0f * tau0, APPROX ? FAST : STL) - 3.0 * tau0) - rval;
}

inline double tauPrimeUniverse(const double &x, const double &tau0)
{
	return 6.0 * POW2(SINH(1.5f * x, APPROX ? FAST : STL), EXACT) / (SINH(3.0f * tau0, APPROX ? FAST : STL) - 3.0 * tau0);
}

inline double phi4D(const double &x, const double &rval)
{
	return (2.0 * x - SIN(2.0f * x, APPROX ? FAST : STL)) / TWO_PI - rval;
}

inline double phiPrime4D(const double &x)
{
	return POW2(SIN(static_cast<float>(x), APPROX ? FAST : STL), EXACT) / HALF_PI;
}

//Returns zeta Residual
//Used in 1+1 and 3+1 Causets
inline double solveZeta(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
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
		-1.0f * eta02D(x, *p5, *p4) / eta0Prime2D(x) :
		-1.0f * zeta4D(x, *p5, *p4) / zetaPrime4D(x));
}

//Returns tau0 Residual
//Used in Universe Causet
inline double solveTau0(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		//No null pointers
		assert (p1 != NULL);	//alpha
		assert (p2 != NULL);	//delta
		assert (p3 != NULL);	//a
		assert (p5 != NULL);	//N_tar

		//Variables in the correct ranges
		assert (*p1 > 0.0);
		assert (*p2 > 0.0);
		assert (*p3 > 0.0);
		assert (*p5 > 0);
	}

	return (-1.0f * tau0(x, *p5, *p1, *p2, *p3) / tau0Prime(x));
}

//Returns tau Residual
//Used in 3+1 Causet
inline double solveTau(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		//No null pointers
		assert (p1 != NULL);	//zeta
		assert (p3 != NULL);	//rval

		//Variables in the correct ranges
		assert (*p1 > 0.0 && *p1 < HALF_PI);
		assert (*p3 > 0.0 && *p3 < 1.0);
	}

	return (-1.0f * tau4D(x, *p1, *p3) / tauPrime4D(x, *p1));
}

//Returns tau Residual
//Used in Universe Causet
inline double solveTauUniverse(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		//No null pointers
		assert (p1 != NULL);	//tau0
		assert (p2 != NULL);	//rval

		//Variables in the correct ranges
		assert (*p1 > 0.0);
		assert (*p2 > 0.0 && *p2 < 1.0);
	}

	return (-1.0f * tauUniverse(x, *p1, *p2) / tauPrimeUniverse(x, *p1));
}

//Returns tau Residual in Bisection Algorithm
//Used in Universe Causet
inline double solveTauUnivBisec(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		assert (p1 != NULL);	//tau0
		assert (p2 != NULL);	//rval

		assert (*p1 > 0.0);
		assert (*p2 > 0.0 && *p2 < 1.0);
	}

	return tauUniverse(x, *p1, *p2);
}

//Returns phi Residual
//Used in 3+1 and Universe Causets
inline double solvePhi(const double &x, const double * const p1, const double * const p2, const double * const p3, const float * const p4, const int * const p5, const int * const p6)
{
	if (DEBUG) {
		//No null pointers
		assert (p1 != NULL);	//rval

		//Variables in correct ranges
		assert (*p1 > 0.0 && *p1 < 1.0);
	}

	return (-1.0f * phi4D(x, *p1) / phiPrime4D(x));
}

//Math Functions for Gauss Hypergeometric Function

//This is used to solve for a more exact solution than the one provided
//by numerical integration using the tauToEtaUniverse function
inline float _2F1_tau(const float &tau, void * const param)
{
	return 1.0f / POW2(COSH(1.5f * tau, APPROX ? FAST : STL), EXACT);
}

//This is used to evaluate xi(r) in the rescaledDegreeUniverse
//function for r > 1
inline float _2F1_r(const float &r, void * const param)
{
	return -1.0f / POW3(r, EXACT);
}

//De Sitter Spatial Lengths

//X1 Coordinate of de Sitter Metric
inline float X1(const float &phi)
{
	return COS(phi, APPROX ? FAST : STL);
}

//X2 Coordinate of de Sitter Metric
inline float X2(const float &phi, const float &chi)
{
	return SIN(phi, APPROX ? FAST : STL) * COS(chi, APPROX ? FAST : STL);
}

//X3 Coordinate of de Sitter Metric
inline float X3(const float &phi, const float &chi, const float &theta)
{
	return SIN(phi, APPROX ? FAST : STL) * SIN(chi, APPROX ? FAST : STL) * COS(theta, APPROX ? FAST : STL);
}

//X4 Coordinate of de Sitter Metric
inline float X4(const float &phi, const float &chi, const float &theta)
{
	return SIN(phi, APPROX ? FAST : STL) * SIN(chi, APPROX ? FAST : STL) * SIN(theta, APPROX ? FAST : STL);
}

//Spherical Inner Product
inline float sphProduct(const float4 &sc0, const float4 &sc1)
{
	return X1(sc0.y) * X1(sc1.y) +
	       X2(sc0.y, sc0.z) * X2(sc1.y, sc1.z) +
	       X3(sc0.y, sc0.z, sc0.x) * X3(sc1.y, sc1.z, sc1.x) +
	       X4(sc0.y, sc0.z, sc0.x) * X4(sc1.y, sc1.z, sc1.x);
}

//Temporal Transformations

//Conformal to Rescaled Time
inline float etaToTau(const float eta)
{
	return ACOSH(1.0f / COS(eta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Rescaled to Conformal Time
inline float tauToEta(const float tau)
{
	return ACOS(1.0f / COSH(tau, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
}

//Minkowski to Conformal Time (Universe)

//For use with GNU Scientific Library
inline double tauToEtaUniverse(double tau, void *params)
{
	return POW(SINH(1.5f * tau, APPROX ? FAST : STL), (-2.0f / 3.0f), APPROX ? FAST : STL);
}

//Exact Solution (no integration)
inline float tauToEtaUniverseExact(const float &tau, const float &a, const float &alpha)
{
	float z = _2F1_tau(tau, NULL);
	float eta, f, err;
	int nterms = 50;

	_2F1(&_2F1_tau, tau, NULL, 1.0f / 3.0f, 5.0f / 6.0f, 4.0f / 3.0f, &f, &err, &nterms);

	eta = 3.0f * GAMMA(-1.0f / 3.0f, STL) * POW(z, 1.0f / 3.0f, APPROX ? FAST : STL) * f;
	eta += 4.0f * SQRT(3.0f, STL) * POW(M_PI, 1.5f, APPROX ? FAST : STL) / GAMMA(5.0f / 6.0f, STL);
	eta *= a / (9.0f * alpha * GAMMA(2.0f / 3.0f, STL));

	return eta;
}

//Rescaled Average Degree in Universe Causet

//Approximates (108) in [2]
inline double xi(double &r)
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

	return _xi;
}

//Note to get the rescaled averge degree this result must still be
//multiplied by 8pi/(sinh(3tau0)-3tau0)
inline double rescaledDegreeUniverse(int dim, double x[])
{
	//Identify x[0] with x coordinate
	//Identify x[1] with r coordinate

	double z;

	z = POW3(ABS(xi(x[0]) - xi(x[1]), STL), EXACT) * POW2(x[0], EXACT) * POW3(x[1], EXACT) * SQRT(x[1], STL);
	z /= (SQRT(1.0 + 1.0 / POW3(x[0], EXACT), STL) * SQRT(1.0 + POW3(x[1], EXACT), STL));

	return z;
}

//Geodesic Distances

//Embedded Z1 Coordinate
//Used to calculate geodesic distances in universe
//For use with GNU Scientific Library
inline double embeddedZ1(double x, void *params)
{
	GSL_EmbeddedZ1_Parameters *p = (GSL_EmbeddedZ1_Parameters*)params;
	double a = p->a;
	double alpha = p->alpha;

	return SQRT(1.0 + POW2(static_cast<float>(a), EXACT) * x * POW2(static_cast<float>(alpha), EXACT) / (POW3(static_cast<float>(alpha), EXACT) + POW3(static_cast<float>(x), EXACT)), STL);
}

//Returns the de Sitter distance between two nodes
//Modify this to handle 3+1 DS without matter!
//O(xxx) Efficiency (revise this)
inline float distanceDS(EVData *evd, const float4 &node_a, const float &tau_a, const float4 &node_b, const float &tau_b, const int &dim, const Manifold &manifold, const double &a, const double &alpha, const bool &universe)
{
	if (DEBUG) {
		//Parameters in Correct Ranges
		assert (dim == 3);
		assert (manifold == DE_SITTER);
		assert (a > 0.0);
		if (universe)
			assert (alpha > 0.0);
	}

	//Check if they are the same node
	if (node_a.w == node_b.w && node_a.x == node_b.x && node_a.y == node_b.y && node_a.z == node_b.z)
		return 0.0f;

	float z0_a, z0_b;
	float z1_a, z1_b;
	float inner_product_a, inner_product_b, inner_product_ab;
	float signature;
	float distance;

	if (universe) {
		IntData idata = IntData();
		idata.tol = 1e-5;

		GSL_EmbeddedZ1_Parameters p;
		p.a = a;
		p.alpha = alpha;

		//Solve for z1 in Rotated Plane
		float power = 2.0f / 3.0f;
		z1_a = alpha * POW(SINH(1.5f * tau_a, APPROX ? FAST : STL), power, APPROX ? FAST : STL);
		z1_b = alpha * POW(SINH(1.5f * tau_b, APPROX ? FAST : STL), power, APPROX ? FAST : STL);

		//Use Numerical Integration for z0
		idata.upper = z1_a;
		z0_a = integrate1D(&embeddedZ1, (void*)&p, &idata, QNG);
		idata.upper = z1_b;
		z0_b = integrate1D(&embeddedZ1, (void*)&p, &idata, QNG);
	} else {
		z0_a = a * SINH(tau_a, APPROX ? FAST : STL);
		z0_b = a * SINH(tau_b, APPROX ? FAST : STL);

		z1_a = a * COSH(tau_a, APPROX ? FAST : STL);
		z1_b = a * COSH(tau_b, APPROX ? FAST : STL);
	}

	inner_product_a = POW2(z1_a, EXACT) * sphProduct(node_a, node_a) - POW2(z0_a, EXACT);
	inner_product_b = POW2(z1_b, EXACT) * sphProduct(node_b, node_b) - POW2(z0_b, EXACT);
	inner_product_ab = z1_a * z1_b * sphProduct(node_a, node_b) - z0_a * z0_b;
	signature = inner_product_a + inner_product_b - 2.0f * inner_product_ab;

	if (signature < 0.0f)
		//Timelike
		distance = ACOSH(inner_product_ab, APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
	else if (inner_product_ab <= -1.0f)
		//Disconnected Regions
		distance = INF;
	else
		//Spacelike
		distance = ACOS(inner_product_ab, APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);

	//Check light cone condition for 4D vs 5D
	//Null hypothesis is the nodes are not connected
	if (evd != NULL) {
		float d_eta = ABS(node_b.w - node_a.w, STL);
		float d_theta = ACOS(sphProduct(node_a, node_b), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);

		if (signature == 0.0f)
			return distance;

		if (d_theta < d_eta) {	//Actual Time-Like
			if (signature < 0)
				//False Negative (both timelike)
				evd->confusion[1] += 1.0;
			else {
				//True Negative
				evd->confusion[2] += 1.0;
				evd->tn[++evd->tn_idx] = d_eta;
				evd->tn[++evd->tn_idx] = d_theta;
			}
		} else {	//Actual Space-Like
			if (signature < 0) {
				//False Positive
				evd->confusion[3] += 1.0;
				evd->fp[++evd->fp_idx] = d_eta;
				evd->tn[++evd->fp_idx] = d_theta;
			} else
				//True Positive (both spacelike)
				evd->confusion[0] += 1.0;
		}
	}	

	return distance;
}

//Returns the hyperbolic distance between two nodes
//O(xxx) Efficiency (revise this)
inline float distanceH(const float2 &hc_a, const float2 &hc_b, const int &dim, const Manifold &manifold, const double &zeta)
{
	if (DEBUG) {
		assert (dim == 1);
		assert (manifold == HYPERBOLIC);
		assert (zeta != 0.0);
	}

	if (hc_a.x == hc_b.x && hc_a.y == hc_b.y)
		return 0.0f;

	float dtheta = M_PI - ABS(M_PI - ABS(hc_a.y - hc_b.y, STL), STL);
	float distance = ACOSH(COSH(zeta * hc_a.x, APPROX ? FAST : STL) * COSH(zeta * hc_b.x, APPROX ? FAST : STL) - SINH(zeta * hc_a.x, APPROX ? FAST : STL) * SINH(zeta * hc_b.x, APPROX ? FAST : STL) * COS(dtheta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION) / zeta;

	return distance;
}

#endif
