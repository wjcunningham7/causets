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

//BEGIN COMPACT EQUATIONS

inline double tau0(const double &x, const int &N_tar, const double &alpha, const double &delta, const double &a)
{
	return SINH(3.0 * x, APPROX ? FAST : STL) - 3.0 * (x + static_cast<double>(N_tar) / (POW2(M_PI, EXACT) * delta * a * POW(alpha, 3.0, EXACT)));
}

inline double tau0Prime(const double &x)
{
	return 3.0 * (COSH(3.0 * x, APPROX ? FAST : STL) - 1.0);
}

//END COMPACT EQUATIONS

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

inline double phi4D(const double &x, const double &rval)
{
	return (2.0 * x - SIN(2.0 * x, APPROX ? FAST : STL)) / TWO_PI - rval;
}

inline double phiPrime4D(const double &x)
{
	return POW2(SIN(x, APPROX ? FAST : STL), EXACT) / HALF_PI;
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
		-1.0 * eta02D(x, *p5, *p4) / eta0Prime2D(x) :
		-1.0 * zeta4D(x, *p5, *p4) / zetaPrime4D(x));
}

//BEGIN COMPACT EQUATIONS

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

	return (-1.0 * tau0(x, *p5, *p1, *p2, *p3) / tau0Prime(x));
}

//END COMPACT EQUATIONS

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

	return (-1.0 * tau4D(x, *p1, *p3) / tauPrime4D(x, *p1));
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

	return (-1.0 * tauUniverse(x, *p1, *p2) / tauPrimeUniverse(x, *p1));
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

	return (-1.0 * phi4D(x, *p1) / phiPrime4D(x));
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

//BEGIN COMPACT EQUATIONS

//De Sitter Spatial Lengths

//X1 Coordinate of de Sitter Metric
inline float X1(const float &phi)
{
	return static_cast<float>(COS(phi, APPROX ? FAST : STL));
}

//X2 Coordinate of de Sitter Metric
inline float X2(const float &phi, const float &chi)
{
	return static_cast<float>(SIN(phi, APPROX ? FAST : STL) * COS(chi, APPROX ? FAST : STL));
}

//X3 Coordinate of de Sitter Metric
inline float X3(const float &phi, const float &chi, const float &theta)
{
	return static_cast<float>(SIN(phi, APPROX ? FAST : STL) * SIN(chi, APPROX ? FAST : STL) * COS(theta, APPROX ? FAST : STL));
}

//X4 Coordinate of de Sitter Metric
inline float X4(const float &phi, const float &chi, const float &theta)
{
	return static_cast<float>(SIN(phi, APPROX ? FAST : STL) * SIN(chi, APPROX ? FAST : STL) * SIN(theta, APPROX ? FAST : STL));
}

//Spherical Inner Product
inline float sphProduct(const float4 &sc0, const float4 &sc1)
{
	return X1(sc0.y) * X1(sc1.y) +
	       X2(sc0.y, sc0.z) * X2(sc1.y, sc1.z) +
	       X3(sc0.y, sc0.z, sc0.x) * X3(sc1.y, sc1.z, sc1.x) +
	       X4(sc0.y, sc0.z, sc0.x) * X4(sc1.y, sc1.z, sc1.x);
}

//END COMPACT EQUATIONS

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
		//Parameters in Correct Ranges
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
		//Parameters in Correct Ranges
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
		//Variables in Correct Ranges
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
		//No Null Pointers
		assert (table != NULL);

		//Variables in Correct Ranges
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
		//No Null Pointers
		assert (params != NULL);

		//Variables in Correct Ranges
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
		//No Null Pointers
		assert (params != NULL);

		//Variables in Correct Ranges
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
		//No Null Pointers
		assert (params != NULL);

		//Variables in Correct Ranges
		assert (eta > 0.0);
	}

	//Identify params[0] with eta_m
	//Identify params[1] with a
	//Identify params[2] with alpha
	//Identify params[3] with size
	//Identify params[4] with table
	
	return POW3(ABS(((double*)params)[0] - eta, STL), EXACT) * POW2(POW2(rescaledScaleFactor(&((double*)params)[4], ((double*)params)[3], eta, ((double*)params)[1], ((double*)params)[2]), EXACT), EXACT);
}

//BEGIN COMPACT EQUATIONS

//Geodesic Distances

//Embedded Z1 Coordinate
//Used to calculate geodesic distances in universe
//For use with GNU Scientific Library
inline double embeddedZ1(double x, void *params)
{
	if (DEBUG) {
		//No Null Pointers
		assert (params != NULL);
	}

	GSL_EmbeddedZ1_Parameters *p = (GSL_EmbeddedZ1_Parameters*)params;

	double a = p->a;
	double alpha = p->alpha;
	
	if (DEBUG) {
		//Variables in Correct Ranges
		assert (a > 0.0);
		assert (alpha > 0.0);
	}

	return SQRT(1.0 + POW2(a, EXACT) * x * POW2(alpha, EXACT) / (POW3(alpha, EXACT) + POW3(x, EXACT)), STL);
}

//Returns the de Sitter distance between two nodes
//Modify this to handle 3+1 DS without matter!
//O(xxx) Efficiency (revise this)
inline double distanceDS(EVData *evd, const float4 &node_a, const float &tau_a, const float4 &node_b, const float &tau_b, const int &dim, const Manifold &manifold, const double &a, const double &alpha, const bool &universe)
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
		return 0.0;

	double z0_a, z0_b;
	double z1_a, z1_b;
	double inner_product_a, inner_product_b, inner_product_ab;
	double signature;
	double distance;

	if (universe) {
		IntData idata = IntData();
		idata.tol = 1e-5;

		GSL_EmbeddedZ1_Parameters p;
		p.a = a;
		p.alpha = alpha;

		//Solve for z1 in Rotated Plane
		double power = 2.0 / 3.0;
		z1_a = alpha * POW(SINH(1.5 * tau_a, APPROX ? FAST : STL), power, APPROX ? FAST : STL);
		z1_b = alpha * POW(SINH(1.5 * tau_b, APPROX ? FAST : STL), power, APPROX ? FAST : STL);

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
	signature = inner_product_a + inner_product_b - 2.0 * inner_product_ab;

	if (signature < 0.0)
		//Timelike
		distance = ACOSH(inner_product_ab, APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);
	else if (inner_product_ab <= -1.0)
		//Disconnected Regions
		distance = INF;
	else
		//Spacelike
		distance = ACOS(inner_product_ab, APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);

	//Check light cone condition for 4D vs 5D
	//Null hypothesis is the nodes are not connected
	if (evd != NULL) {
		double d_eta = ABS(static_cast<double>(node_b.w - node_a.w), STL);
		double d_theta = ACOS(static_cast<double>(sphProduct(node_a, node_b)), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION);

		if (signature == 0.0)
			return distance;

		if (d_theta < d_eta) {	//Actual Time-Like
			if (signature < 0)
				//False Negative (both timelike)
				evd->confusion[1] += 1.0;
			else {
				//True Negative
				evd->confusion[2] += 1.0;
				evd->tn[evd->tn_idx++] = static_cast<float>(d_eta);
				evd->tn[evd->tn_idx++] = static_cast<float>(d_theta);
			}
		} else {	//Actual Space-Like
			if (signature < 0) {
				//False Positive
				evd->confusion[3] += 1.0;
				evd->fp[evd->fp_idx++] = static_cast<float>(d_eta);
				evd->tn[evd->fp_idx++] = static_cast<float>(d_theta);
			} else
				//True Positive (both spacelike)
				evd->confusion[0] += 1.0;
		}
	}	

	return distance;
}

//END COMPACT EQUATIONS

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
