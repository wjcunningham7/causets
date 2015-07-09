#ifndef GEODESICS_H_
#define GEODESICS_H_

#include "Causet.h"
#include "Operations.h"

/////////////////////////////
//(C) Will Cunningham 2015 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//================================================//
// Approximtions to omega = f(tau1, tau2, lambda) //
//================================================//

//----------//
// Region 1 //
//----------//

inline double omegaRegion1(const double &x, const double &lambda, double * const err, int * const nterms)
{
	if (DEBUG) {
		assert (err != NULL);
		assert (nterms != NULL);
		assert (x >= 0.0 && x <= GEODESIC_LOWER);
		assert (lambda != 0.0);
		assert (*err >= 0.0);
		assert (!(*err == 0.0 && *nterms == -1));
	}

	double omega = 0.0;

	//Used for _2F1
	double f;
	double f_err;
	int f_nt;

	//Determine which transformation of 2F1 is used
	double z = -1.0 * lambda * POW2(POW2(x, EXACT), EXACT);
	//printf("z: %f\n", z);
	double w, w1;
	int method = 0;
	if (z >= 0.0 && z <= 0.5) {
		w = z;
		method = 0;
	} else if (z > 0.5 && z <= 1.0) {
		w = 1.0 - z;
		w1 = SQRT(w, APPROX ? FAST : STL);
		method = 1;
	} else if (z >= -1.0 && z < 0.0) {
		w = z / (z - 1.0);
		method = 2;
	} else if (z < -1.0) {
		w = 1.0 / (1.0 - z);
		w1 = 1.0 / w;
		method = 3;
	} else
		// This should never be reached
		return NAN;
	//printf("w: %f\n", w);

	//Series solution (see notes)
	double error = INF;
	int k = 0;
	double omega_k, k1, k2;
	while (error > *err && k < *nterms) {
		f = 0.0;
		f_err = 1.0E-10;
		f_nt = -1;

		omega_k = 0.0;
		k1 = 1.5 * k;
		k2 = 6.0 * k + 1.0;

		switch (method) {
		case 0:
			// 0 <= z <= 0.5
			//printf("CASE 0.\n");
			_2F1(0.5, k1 + 0.25, k1 + 1.25, w, &f, &f_err, &f_nt, false);
			omega_k = f;
			break;
		case 1:
			// 0.5 < z <= 1
			//printf("CASE 1.\n");
			omega_k = SQRT_PI * GAMMA(k1 + 1.25, STL) * POW(z, -1.0 * (k1 + 0.25), APPROX ? FAST : STL) / GAMMA(k1 + 0.75, STL);
			_2F1(k1 + 0.75, 1.0, 1.5, w, &f, &f_err, &f_nt, false);
			omega_k -= k2 * w1 * f / 2.0;
			break;
		case 2:
			// -1 <= z < 0
			//printf("CASE 2.\n");
			_2F1(0.5, 1.0, k1 + 1.25, w, &f, &f_err, &f_nt, false);
			omega_k = f;
			break;
		case 3:
			// z < -1
			//printf("CASE 3.\n");
			omega_k = GAMMA(k1 + 1.25, STL) * GAMMA(0.25 - k1, STL) * POW(w1 - 1.0, -1.0 * (k1 + 0.25), APPROX ? FAST : STL) / SQRT_PI;
			_2F1(0.5, 1.0, 1.25 - k1, w, &f, &f_err, &f_nt, false);
			omega_k += k2 * SQRT(w, STL) * f / (k2 - 2.0);
			break;
		default:
			// This should never be reached
			return NAN;
		}

		omega_k *= POW(x, k2, APPROX ? FAST : STL);
		omega_k /= GAMMA(k + 1, STL) * GAMMA(0.5 - k, STL) * k2;
		//printf("k: %d\tf: %.8e\tomega_k: %.8e\n", k, f, omega_k);

		omega += omega_k;
		error = ABS(omega_k / omega, STL);
		k++;
	}

	if (method == 2)
		omega *= SQRT(1.0 - w, APPROX ? FAST : STL);

	omega *= 2.0 * SQRT_PI;

	*nterms = k;
	*err = error;

	return omega;
}

//----------//
// Region 2 //
//----------//

//Supplementary functions used for elliptic integral approximations
//Described in [4]
inline double theta_nm(const int &n, const int &m)
{
	if (DEBUG) {
		assert (n >= 0);
		assert (m >= 1);
	}

	return static_cast<double>(m * M_PI / (2.0 * n + 1.0));
}

inline double sigma_nm(const int &n, const int &m)
{
	if (DEBUG) {
		assert (n >= 0);
		assert (m >= 1);
	}

	return SQRT(1.0 + POW2(SIN(theta_nm(n, m), APPROX ? FAST : STL), EXACT), STL);
}

inline double rho_nm(const int &n, const int &m)
{
	if (DEBUG) {
		assert (n >= 0);
		assert (m >= 1);
	}

	return SQRT(1.0 + POW2(COS(theta_nm(n, m), APPROX ? FAST : STL), EXACT), STL);
}

//Returns a complex phi for lambda > 0
inline double2 ellipticPhi(const double &x, const double &lambda)
{
	if (DEBUG) {
		assert (x > 0.0);
		assert (lambda > 0.0);
	}

	double2 phi;
	double z = lambda * POW2(POW2(x, EXACT), EXACT);
	double sqz = SQRT(z, STL);

	double a = sqz / 2.0;
	double t1 = SQRT(1.0 + 0.25 / POW2(a, EXACT), STL);
	double t2 = 0.5 / a;
	
	phi.x = -1.0 * ATAN(SQRT(t1 - t2, STL), STL, VERY_HIGH_PRECISION);
	//printf("phiR: %f\n", phi.x);

	if (DEBUG)
		assert (phi.x > -1.0 * HALF_PI / 2.0);

	double t3 = 2.0 * a;
	double t4 = t3 + SQRT(POW2(t3, EXACT) + 1.0, STL);
	double t5 = SQRT(POW2(t4, EXACT) - 1.0, STL);

	phi.y = 0.5 * LOG(t4 + t5, APPROX ? FAST : STL);
	//printf("phiI: %f\n", phi.y);

	if (DEBUG)
		assert (phi.y >= 0.0);

	return phi;
}

//xi(phi, tau_m) = arctan(tau_m * tan(phi))
inline double2 ellipticXi(const double2 &upsilon, const double &tau_nm)
{
	if (DEBUG) {
		assert (upsilon.x <= 0.0);
		assert (upsilon.y >= 0.0 && upsilon.y < 1.0);
		assert (tau_nm > 0.0);
	}

	double2 xi;
	double a = upsilon.y;
	double b = upsilon.x;
	double c = tau_nm;

	double ac2 = POW2(a * c, EXACT);
	double bc2 = POW2(b * c, EXACT);
	double ac2_bc2 = ac2 + bc2;

	xi.x = -0.5 * ATAN(2.0 * b * c / (ac2_bc2 - 1.0), STL, VERY_HIGH_PRECISION);
	if (xi.x > 0.0)
		//This results from a +/- in the numerical solution
		xi.x -= HALF_PI;
	xi.y = 0.25 * LOG((ac2_bc2 + 2.0 * a * c + 1.0) / (ac2_bc2 - 2.0 * a * c + 1.0), STL);

	if (DEBUG) {
		assert (xi.x < 0.0);
		assert (xi.y > 0.0);
	}

	return xi;
}

//Approximates the incomplete elliptic integral of the first kind with k^2 = -1
//This form assumes phi is real-valued
inline double ellipticIntF(const double &phi, const int &n)
{
	if (DEBUG) {
		//assert (phi % HALF_PI != 0.0);
		assert (n >= 0);
	}

	double f = 0.0;
	double s;
	int m;

	for (m = 1; m <= n; m++) {
		s = sigma_nm(n, m);
		f += s * ATAN(TAN(phi, APPROX ? FAST : STL), STL, VERY_HIGH_PRECISION) / s;
	}

	f *= 2.0;
	f += phi;
	f /= (2.0 * n + 1.0);

	return f;
}

//Approximates the incomplete elliptic integral of the first kind with k^2 = -1
//This form assumes phi is complex-valued with
//phi.x = Re(phi) and phi.y = Im(phi)
//This function returns Re[(1 \pm i) * F_n(phi, i)]
inline double ellipticIntF_Complex(const double2 &phi, const int &n, const bool &plus)
{
	if (DEBUG) {
		assert (phi.x > -0.25 * M_PI && phi.x <= 0.0);
		assert (phi.y >= 0.0);
		assert (n >= 0);
	}

	double2 upsilon;
	double2 xi;

	double f = 0.0;
	double u_norm;
	double s;
	int m;

	u_norm = COS(2.0 * phi.x, APPROX ? FAST : STL) + COSH(2.0 * phi.y, APPROX ? FAST : STL);
	upsilon.x = SIN(2.0 * phi.x, APPROX ? FAST : STL) / u_norm;
	upsilon.y = SINH(2.0 * phi.y, APPROX ? FAST : STL) / u_norm;
	//printf("uR: %f\n", upsilon.x);
	//printf("uI: %f\n", upsilon.y);

	for (m = 1; m <= n; m++) {
		s = sigma_nm(n, m);
		xi = ellipticXi(upsilon, s);
		//printf("%d\t%f\t%f\n", m, xi.x, xi.y);
		f += (plus ? (xi.x - xi.y) : (xi.x + xi.y)) / s;
	}

	f *= 2.0;
	f += plus ? (phi.x - phi.y) : (phi.x + phi.y);
	f /= (2.0 * n + 1.0);

	return f;
}

//Approximates the incomplete elliptic integral of the second kind with k^2 = -1
//This form assumes phi is real-valued
inline double ellipticIntE(const double &phi, const int &n)
{
	if (DEBUG) {
		//assert (phi % HALF_PI != 0.0);
		assert (n >= 0);
	}

	double e = 0.0;
	double t, r;
	int m;

	for (m = 1; m <= n; m++) {
		t = theta_nm(n, m);
		r = rho_nm(n, m);
		e += ATAN(r * TAN(phi, APPROX ? FAST : STL), STL, VERY_HIGH_PRECISION) * POW2(TAN(t, APPROX ? FAST : STL), EXACT) / r;
	}

	double n2 = 2.0 * n + 1;
	e *= 2.0 / n2;

	return n2 * phi - e; 
}

//Approximates the incomplete elliptic integral of the second kind with k^2 = -1
//This form assumes phi is complex-valued with
//phi.x = Re(phi) and phi.y = Im(phi)
//This function returns Re[(1 \pm i) * E_n(phi, i)]
inline double ellipticIntE_Complex(const double2 &phi, const int &n, const bool &plus)
{
	if (DEBUG) {
		assert (phi.x > -0.25 * M_PI && phi.x <= 0.0);
		assert (phi.y >= 0.0);
		assert (n >= 0);
	}

	double2 upsilon;
	double2 xi;

	double e = 0.0;
	double u_norm;
	double r;
	int m;

	u_norm = COS(2.0 * phi.x, APPROX ? FAST : STL) + COSH(2.0 * phi.y, APPROX ? FAST : STL);
	upsilon.x = SIN(2.0 * phi.x, APPROX ? FAST : STL) / u_norm;
	upsilon.y = SINH(2.0 * phi.y, APPROX ? FAST : STL) / u_norm;

	for (m = 1; m <= n; m++) {
		r = rho_nm(n, m);
		xi = ellipticXi(upsilon, r);
		e += (plus ? (xi.x - xi.y) : (xi.x + xi.y)) * POW2(TAN(theta_nm(n, m), APPROX ? FAST : STL), EXACT) / r;
	}

	double n2 = 2.0 * n + 1.0;
	e *= 2.0 / n2;
	
	return n2 * (plus ? (phi.x - phi.y) : (phi.x + phi.y)) - e;
}

//Region 2, Positive Lambda
//Revise this later to simplify elliptic integral calculations
inline double omegaRegion2a(const double &x, const double &lambda, double * const err, int * const nterms)
{
	if (DEBUG) {
		assert (err != NULL);
		assert (nterms != NULL);
		assert (x >= GEODESIC_LOWER && x <= GEODESIC_UPPER);
		assert (lambda > 0.0);
		assert (*err >= 0.0);
		assert (!(*err == 0.0 && *nterms == -1));
	}

	double2 phi = ellipticPhi(x, lambda);
	double elF_plus = ellipticIntF_Complex(phi, *nterms, true);
	double elF_minus = ellipticIntF_Complex(phi, *nterms, false);
	double elE_minus = ellipticIntE_Complex(phi, *nterms, false);
	double sql = SQRT(lambda, STL);

	//printf("elF_plus:\t%.8e\n", elF_plus);
	//printf("elF_minus:\t%.8e\n", elF_minus);
	//printf("elE_minus:\t%.8e\n", elE_minus);

	//double omega = (55.0 * SQRT(1.0 + lambda * POW2(POW2(x, EXACT), EXACT), STL) + 153.0 * sql * ASINH(sql * POW2(x, EXACT), STL, VERY_HIGH_PRECISION)) / (16.0 * SQRT(2.0, STL) * lambda);
	//omega += 21.0 * elF_plus / (16.0 * SQRT(sql, STL));
	//omega -= 171.0 * (elE_minus - elF_minus) / (16.0 * POW(lambda, 0.75, STL));
	double omega = 0.0;
	double t1 = (55.0 * SQRT(1.0 + lambda * POW2(POW2(x, EXACT), EXACT), STL) + 153.0 * sql * ASINH(sql * POW2(x, EXACT), STL, VERY_HIGH_PRECISION)) / (16.0 * SQRT(2.0, STL) * lambda);
	double t2 = 21.0 * elF_plus / (16.0 * SQRT(sql, STL));
	double t3 = 171.0 * (elE_minus - elF_minus) / (16.0 * POW(lambda, 0.75, STL));
	omega = t1 + t2 - t3;

	//printf("t1:\t%.8e\n", t1);
	//printf("t2:\t%.8e\n", t2);
	//printf("t3:\t%.8e\n", t3);

	return omega;
}

//Region 2, Negative Lambda
inline double omegaRegion2b(const double &x, const double &lambda, double * const err, int * const nterms)
{
	if (DEBUG) {
		assert (err != NULL);
		assert (nterms != NULL);
		assert (x >= GEODESIC_LOWER && x <= GEODESIC_UPPER);
		assert (lambda < 0.0);
		assert (*err >= 0.0);
		assert (!(*err == 0.0 && *nterms == -1));
	}

	double phi = asin(POW(-1.0 * lambda, 0.25, APPROX ? FAST : STL) * x);
	double el_F = ellipticIntF(phi, *nterms);
	double el_E = ellipticIntE(phi, *nterms);
	double sql = SQRT(-1.0 * lambda, STL);

	double omega = (55.0 * SQRT(1.0 + lambda * POW2(POW2(x, EXACT), EXACT), STL) - 153.0 * sql * asin(sql * POW2(x, EXACT))) / (16.0 * SQRT(2.0, STL) * lambda);
	omega -= 21.0 * el_F / (8.0 * SQRT(2.0 * sql, STL));
	omega -= 171.0 * (el_F - el_E) / (8.0 * SQRT(2.0 * POW3(sql, EXACT), STL));

	return omega;
}

//----------//
// Region 3 //
//----------//

//Supplementary functions for Region 3
//See notes for how/why they are used

inline double ln_hnl(const double &x, const double &lambda, const int &l, const int &n)
{
	if (DEBUG) {
		assert (x >= GEODESIC_UPPER);
		assert (lambda != 0.0);
		assert (l >= 0 && !(l % 2));
		assert (n >= 0 && n <= (3 * l) / 2);
	}

	double t1 = - 1.5 * static_cast<double>(l) - 0.5;
	double t2 = static_cast<double>(n) + t1;
	double t3 = 1.0 + lambda * POW2(POW2(x, EXACT), EXACT);

	return LOG(GAMMA(t2, STL) * SIN(t2 * M_PI, APPROX ? FAST : STL), APPROX ? FAST : STL) - LOG(GAMMA(t1, STL) * SIN(t1 * M_PI, APPROX ? FAST : STL), APPROX ? FAST : STL) - LOGGAMMA(static_cast<double>(n) + 1.0, STL) - static_cast<double>(n) * LOG(t3, APPROX ? FAST : STL);
}

inline double ln_fl(const double &x, const double &lambda, const int &l)
{
	if (DEBUG) {
		assert (x >= GEODESIC_UPPER);
		assert (lambda != 0.0);
		assert (l >= 0 && !(l % 2));
	}

	double t1 = LOGGAMMA(1.5 * static_cast<double>(l) + 1.0 + 0.000001, STL);
	double t2 = - 1.5 * static_cast<double>(l) - 0.5;
	double sum = 0.0;
	double t3;

	for (int n = 1; n <= 3 * l / 2; n++) {
		t3 = static_cast<double>(n) + t1;
		sum += exp(ln_hnl(x, lambda, l, n)) / SIN(t3 * M_PI, APPROX ? FAST : STL);
	}
	sum *= SIN(t2 * M_PI, APPROX ? FAST : STL);

	return t1 + LOG(1.0 + sum, APPROX ? FAST : STL);
}

//Must multiply by csc((1-3l)pi/2) after taking e^(ln_Fl)
inline double ln_Fl(const double &x, const double &lambda, const int &l)
{
	if (DEBUG) {
		assert (x >= GEODESIC_UPPER);
		assert (lambda != 0.0);
		assert (l >= 0 && !(l % 2));
	}

	double t1 = -1.0 * LOG(SQRT_PI, STL);
	double t2 = -1.5 * static_cast<double>(l) + 0.5;
	double t3 = 1.0 + lambda * POW2(POW2(x, EXACT), EXACT);

	double t4 = LOG(GAMMA(t2, STL) * SIN(t2 * M_PI, APPROX ? FAST : STL), APPROX ? FAST : STL);
	double t5 = (1 - t2) * LOG(t3, APPROX ? FAST : STL);

	return t1 + t4 + t5 + ln_fl(x, lambda, l);
}

//Calculates the even terms in the series for x, lambda
inline double omegaRegion3a(const double &x, const double &lambda, double * const err, int * const nterms)
{
	return 0.0;
}

//Calculates the even terms in the series for x1, x2, lambda
//Note this is used when lambda is large and the Hypergeometric function diverges
inline double omegaRegion3aDiv(const double &x1, const double &x2, const double &lambda, double * const err, int * const nterms)
{
	return 0.0;
}

//Calculates the odd terms in the series for x, lambda
inline double omegaRegion3b(const double * const table, const double &x, const double &lambda, double * const err, int * const nterms, const long &size)
{
	return 0.0;
}

//Region 3
inline double omegaRegion3(const double * const table, const double &x1, const double &x2, const double &lambda, double * const err, int * const nterms, const long &size)
{
	return 0.0;
}

//Region 3
/*inline double omegaRegion3(const double * const table, const double &x, const double &lambda, double * const err, int * const nterms, const long &size)
{
	if (DEBUG) {
		assert (table != NULL);
		assert (err != NULL);
		assert (nterms != NULL);
		assert (x >= GEODESIC_UPPER);
		assert (lambda != 0.0);
		//assert (lambda < 1.0);	//This prevents the series from diverging
		assert (*err >= 0.0);
		assert (!(*err == 0.0 && *nterms == -1));
		assert (size > 0L);
		assert (*nterms <= 100);	//Limit of the lookup table
	}

	double omega = 0.0;
	int p;

	//Used for _2F1
	double f;
	double f_err;
	int f_nt;

	//Determine which transformation of 2F1 is used for the even terms
	double z = -1.0 * lambda * POW2(POW2(x, EXACT), EXACT);
	double w, w1;
	int method = 0;
	if (z >= 0.0 && z <= 0.5) {
		w = z;
		method = 0;
	} else if (z > 0.5 && z <= 1.0) {
		w = 1.0 - z;
		w1 = SQRT(w, APPROX ? FAST : STL);
		method = 1;
	} else if (z >= -1.0 && z < 0.0) {
		w = z / (z - 1.0);
		w1 = 1.0 / SQRT(1.0 - z, APPROX ? FAST : STL);
		method = 2;
	} else if (z < -1.0) {
		w = 1.0 / (1.0 - z);
		w1 = 0.000001;
		method = 3;
	} else
		// This should never be reached
		return NAN;

	//Calculate the even terms in the series
	double error_l = INF;
	int l = 0;
	double omega_l, l1, l2;
	while (error_l > *err && l < *nterms) {
		f = 0.0;
		f_err = 1.0E-10;
		f_nt = -1;

		omega_l = 0.0;
		l1 = 1.5 * l;
		l2 = 6.0 * l + 2.0;

		switch (method) {
		case 0:
			// 0 <= z <= 0.5
			//printf("CASE 0.\n");
			_2F1(0.5, -1.0 * l1 - 0.5, 0.5 - l1, w, &f, &f_err, &f_nt, false);
			omega_l = f;
			break;
		case 1:
			// 0.5 < z <= 1
			//printf("CASE 1.\n");
			_2F1(-1.0 * l1, 1.0, 1.5, w, &f, &f_err, &f_nt, false);
			omega_l = l2 * w1 * f / 2.0;
			break;
		case 2:
			// -1 <= z < 0
			//printf("CASE 2.\n");
			_2F1(0.5, 1.0, 0.5 - l1, w, &f, &f_err, &f_nt, false);
			omega_l = w1 * f;
			break;
		case 3:
			// z < -1
			//printf("CASE 3.\n");
			for (p = 0; p <= l1; p++)
				omega_l += POW(-1.0, p, STL) * POCHHAMMER(-1.0 * l1 - 0.5, p) * POCHHAMMER(-1.0 * l1, p) * GAMMA(l1 + 1.0 - p + w1, STL) * POW(w, p - l1 - 0.5, APPROX ? FAST : STL) / GAMMA(p + 1.0, STL);
			omega_l *= GAMMA(0.5 - l1, STL) / SQRT_PI;
			f = omega_l;	//
			break;
		default:
			// This should never be reached
			return NAN;
		}

		omega_l /= -1.0 * l2 * POW(x, l2, APPROX ? FAST : STL) * GAMMA(l + 1.0, STL) * GAMMA(0.5 - l, STL);
		printf("l: %d\tf: %.8e\tomega_l: %.8e\n", l, f, omega_l);

		omega += omega_l;
		error_l = ABS(omega_l / omega, APPROX ? FAST : STL);
		l += 2;
	}

	//Calculate the odd terms in the series
	z = SQRT(1.0 - z, STL);
	double z1 = z + 1.0;
	double z2 = z - 1.0;
	double error_m = INF;
	int m = 1;

	double omega_m, n1;
	int n, n3, start;
	while (error_m > *err && m < *nterms) {
		omega_m = 0.0;
		n = (m + 1) / 2;
		n3 = 3 * n;
		n1 = 0.5 * POW(lambda, n3 - 1.0, STL);
		start = n3 * (n - 1);

		omega_m += table[start] * LOG(z1, APPROX ? FAST : STL);
		omega_m += table[start+n3] * LOG(ABS(z2, STL), APPROX ? FAST : STL);
		//printf("A1 = %f\tB1 = %f\n", table[start], table[start+n3]);
		for (p = 1; p < n3; p++) {
			omega_m += table[start+p] / (-1.0 * p * POW(z1, p, APPROX ? FAST : STL));
			omega_m += table[start+n3+p] / (-1.0 * p * POW(z2, p, APPROX ? FAST : STL));
			//printf("A%d = %f\tB%d = %f\n", p+1, table[start+p], p+1, table[start+n3+p]);
		}

		omega_m *= n1;
		omega_m /= GAMMA(m + 1.0, STL) * GAMMA(0.5 - m, STL);
		printf("m: %d\tomega_m: %.8e\n", m, omega_m);

		omega += omega_m;
		error_m = ABS(omega_m / omega, APPROX ? FAST : STL);
		m += 2;
	}

	omega *= 2.0 * SQRT_PI;
	
	*nterms = l < m ? l : m;
	*err = error_l < error_m ? error_l : error_m;

	return omega;
}*/

//Gives omega = f(x, lambda) using numerical approximations
//This function is an intermediary subroutine designed to pick the correct region for x
//NOTE: x is related to tau by x = sinh(1.5*tau)^(1/3)
inline double omegaRegionX(const double * const table, const double &x, const double &lambda, double * const err, int * const nterms, const long &size)
{
	if (DEBUG) {
		assert (table != NULL);
		assert (err != NULL);
		assert (nterms != NULL);
		assert (x >= 0.0);
		assert (lambda != 0.0);
		assert (*err >= 0.0);
		assert (!(*err == 0.0 && *nterms == -1));
		assert (size > 0L);
		assert (*nterms <= 100);	//Limit of the lookup table for Region 3
	}

	double omega = -1.0;

	if (x >= GEODESIC_UPPER)
		//Region 3
		omega = omegaRegion3(table, x, lambda, err, nterms, size);
	else if (x >= GEODESIC_LOWER) {
		if (lambda > 0.0)
			//Region 2a
			omega = omegaRegion2a(x, lambda, err, nterms);
		else
			//Region 2b
			omega = omegaRegion2b(x, lambda, err, nterms);
	} else
		//Region 1
		omega = omegaRegion1(x, lambda, err, nterms);

	return omega;
}

//Adds the boundary terms if necessary (see notes)
inline double omegaDelta(const double &x1, const double &x2, const double &lower_delta, const double &upper_delta)
{
	if (DEBUG) {
		assert (x1 >= 0.0);
		assert (x2 >= 0.0);
	}

	double delta = 0.0;
	if (x1 < GEODESIC_LOWER && x2 >= GEODESIC_LOWER)
		delta -= lower_delta;
	if (x1 < GEODESIC_UPPER && x2 >= GEODESIC_UPPER)
		delta -= upper_delta;

	return delta;
}

//Maximum Time in Geodesic (non-embedded)
//Returns tau_max=f(lambda) with lambda < 0
inline double geodesicMaxTau(const Manifold &manifold, const double &lambda)
{
	if (DEBUG)
		assert (manifold == DE_SITTER || manifold == FLRW);

	if (lambda >= 0.0)
		return 0.0f;

	if (manifold == FLRW)
		return (2.0 / 3.0) * ASINH(POW(ABS(lambda, STL), -0.75, STL), STL, VERY_HIGH_PRECISION);
	else if (manifold == DE_SITTER) {
		double g = POW(ABS(lambda, STL), -0.5, STL);
		return g >= 1.0 ? ACOSH(g, STL, VERY_HIGH_PRECISION) : 0.0;
	}

	return 0.0;
}

//Maximum X Time in FLRW Geodesic (non-embedded)
//Returns x_max = f(lambda) with lambda < 0
inline double geodesicMaxX(const double &lambda)
{
	if (DEBUG)
		assert (lambda < 0.0);

	return POW(-1.0 * lambda, -0.25, APPROX ? FAST : STL);
}

//Gives omega12 = f(x1, x2, lambda) using numerical approximations
//This function should be faster than the numerical integral defined in the kernel
//functions, flrwLookupKernel or flrwLookupKernelX
//NOTE: x is related to tau by x = sinh(1.5*tau)^(1/3)
inline double solveOmega12(const double * const table, const double &x1, const double &x2, const double &lambda, double * const err, int * const nterms, const long &size, const double &lower_delta, const double &upper_delta)
{
	if (DEBUG) {
		assert (table != NULL);
		assert (err != NULL);
		assert (nterms != NULL);
		assert (x1 >= 0.0);
		assert (x2 >= 0.0);
		assert (*err >= 0.0);
		assert (!(*err == 0.0 && *nterms == -1));
		assert (size > 0L);
		assert (*nterms <= 100);	//Limit of the lookup table for Region 3
	}

	//Solve for omega(x1, lambda)
	double omega1 = omegaRegionX(table, x1, lambda, err, nterms, size);

	//Solve for omega(x2, lambda)
	double omega2 = omegaRegionX(table, x2, lambda, err, nterms, size);

	double omega;
	if (lambda > 0.0) {
		omega = omega2 - omega1;
		omega += omegaDelta(x1, x2, lower_delta, upper_delta);
	} else if (lambda < 0.0) {
		double x_max = geodesicMaxX(lambda);
		if (x1 > x_max || x2 > x_max)
			return NAN;

		double omega3 = omegaRegionX(table, x_max, lambda, err, nterms, size);
		omega = 2.0 * omega3 - omega1 - omega2;
		omega += omegaDelta(x1, x_max, lower_delta, upper_delta);
		omega += omegaDelta(x2, x_max, lower_delta, upper_delta);
	}

	return omega;
}

//Embedded Z1 Coordinate used in Naive Embedding
//For use with GNU Scientific Library
inline double embeddedZ1(double x, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (x >= 0.0);
	}

	double alpha_tilde = ((double*)params)[0];

	if (DEBUG)
		assert (alpha_tilde != 0.0);

	return SQRT(1.0 + (x / (POW3(alpha_tilde, EXACT) + POW3(x, EXACT))), STL);
}

//Integrands for Exact Geodesic Calculations
//For use with GNU Scientific Library

//Distance Kernels

inline double deSitterDistKernel(double x, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (x >= 0);
	}

	double *p = (double*)params;
	double lambda = p[0];

	if (DEBUG)
		assert (lambda != 0.0);

	double lcx2 = lambda * POW2(COSH(x, STL), EXACT);
	double distance = SQRT(ABS(lcx2 / (1.0 + lcx2), STL), STL);

	return distance;
}

inline double flrwDistKernel(double x, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (x >= 0);
	}

	double *p = (double*)params;
	double lambda = p[0];

	if (DEBUG)
		assert (lambda != 0.0);

	double sx = SINH(1.5 * x, STL);
	double lsx43 = lambda * POW(sx, 4.0 / 3.0, STL);
	double distance = SQRT(ABS(lsx43 / (1.0 + lsx43), STL), STL);

	return distance;
}

//Transcendental Kernels Solving omega12=f(tau1,tau2,lambda)
//These do not use the approximations - they use the exact form of the integrals

inline double flrwLookupKernel(double x, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (x >= 0);
	}

	double *p = (double*)params;
	double lambda = p[0];

	if (DEBUG)
		assert (lambda != 0.0);

	double sx = SINH(1.5 * x, STL);
	double sx43 = POW(sx, 4.0 / 3.0, STL);
	double g = sx43 + lambda * POW2(sx43, EXACT);
	double omega12 = g > 0.0 ? POW(g, -0.5, STL) : 0.0;

	return omega12;
}

//Same as flrwLookupKernel but uses a change of variables
//x = sinh(1.5*tau)^(1/3)
//to make calculations faster
//Multiply by 2 afterwards
inline double flrwLookupKernelX(double x, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (x >= 0.0);
	}

	double lambda = ((double*)params)[0];

	if (DEBUG)
		assert (lambda != 0.0);

	double x4 = POW2(POW2(x, EXACT), EXACT);
	double x6 = x4 * POW2(x, EXACT);

	double g = 1.0 + lambda * x4;
	double omega12 = g > 0.0 ? 1.0 / SQRT(g * (1.0 + x6), STL) : 0.0;

	return omega12;
}

inline double deSitterLookupKernel(double x, void *params)
{
	if (DEBUG) {
		assert (params != NULL);
		assert (x >= 0);
	}

	double *p = (double*)params;
	double lambda = p[0];

	if (DEBUG)
		assert (lambda != 0.0);

	double cx2 = POW2(COSH(x, STL), EXACT);
	double g = cx2 + lambda * POW2(cx2, EXACT);
	double omega12 = g > 0 ? POW(g, -0.5, STL) : 0.0;

	return omega12;
}

//The exact solution in de Sitter (the integral can be solved analytically)
inline double deSitterLookupExact(const double &tau, const double &lambda)
{
	double x = 0.0;
	double g = 0.0;

	if (tau > LOG(MTAU, STL) / 6.0)
		x = exp(2.0 * tau) / 2.0;
	else
		x = COSH(2.0 * tau, STL);
	x += 1.0;
	x *= lambda;

	if (x > -2.0 && x < 0.0) {
		double tol = 1e-5;
		double res;
		int i = 1;
		do {
			res = POW(-1.0 * (1.0 + x), static_cast<double>(i), STL) / i;
			x += res;
			i++;
		} while (ABS(res, STL) > tol);
		g /= 2.0;
	} else
		g = LOG(2.0 + x, STL) / -2.0;

	if (tau > LOG(MTAU, STL) / 6.0)
		g += 2.0 * tau - LOG(2.0, STL);
	else
		g += LOG(SINH(tau, STL), STL);
	g += LOG(2.0, STL) / 2.0;

	return ATAN(exp(g), STL, VERY_HIGH_PRECISION);
}

//Approximation for flrwLookupKernelX
//Assumes lambda is large so (1+lambda*x^4) => lambda*x^4
//Multiply by 2*lambda afterwards
//See how this is called by looking in distance_v2() below
inline double flrwLookupApprox(double x, void *params)
{
	if (DEBUG)
		assert (x >= 0.0);

	double x4 = POW2(POW2(x, EXACT), EXACT);
	double x6 = x4 * POW2(x, EXACT);

	return 1.0 / SQRT((x4 * (1.0 + x6)), STL);
}

//=====================//
// Distance Algorithms //
//=====================//

//Returns the distance between two nodes in the non-compact FLRW manifold
//Version 2 does not use the lookup table for lambda = f(omega12, tau1, tau2)
//O(xxx) Efficiency (revise this)
inline double distance_v2(const double * const table, Coordinates *c, const float * const tau, const int &N_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &chi_max, const double &alpha, const long &size, const bool &compact, const int &past_idx, const int &future_idx, const double &lower_delta, const double &upper_delta)
{
	if (DEBUG) {
		assert (table != NULL);
		assert (c != NULL);
		assert (!c->isNull());
		assert (c->getDim() == 4);
		assert (c->w() != NULL);
		assert (c->x() != NULL);
		assert (c->y() != NULL);
		assert (c->z() != NULL);
		assert (tau != NULL);
		assert (N_tar > 0);
		assert (dim == 3);
		assert (manifold == FLRW);
		assert (a > 0.0);
		assert (HALF_PI - zeta > 0.0);
		assert (chi_max > 0.0);
		assert (alpha > 0.0);
		assert (size > 0L);
		assert (!compact);
		assert (past_idx >= 0 && past_idx < N_tar);
		assert (future_idx >= 0 && future_idx < N_tar);
		assert (past_idx < future_idx);
	}

	IntData idata = IntData();
	idata.limit = 60;
	idata.tol = 1.0e-5;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

	double x1 = POW(SINH(1.5 * tau[past_idx], APPROX ? FAST : STL), 1.0 / 3.0, APPROX ? FAST : STL);
	double x2 = POW(SINH(1.5 * tau[future_idx], APPROX ? FAST : STL), 1.0 / 3.0, APPROX ? FAST : STL);
	double omega12;
	double lambda;

	bool timelike = nodesAreRelated(c, N_tar, dim, manifold, a, zeta, chi_max, alpha, compact, past_idx, future_idx, &omega12);

	if (timelike) {
		//Try approximation for large lambda
		idata.lower = x1;
		idata.upper = x2;
		lambda = omega12 / (2.0 * integrate1D(&flrwLookupApprox, NULL, &idata, QNG));

		double om_res = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&lambda, &idata, QNG);

		printf("Error: %f\n", ABS(om_res - omega12, STL) / omega12);
	}

	//Bisection Method
	double res = 1.0, tol = 1E-3;
	double lower, upper;
	int iter = 0, max_iter = 10000;
	if (timelike) {
		lower = 0.0;
		upper = 10000.0;
	} else {
		lower = -1.0 / POW2(POW2(x1, EXACT), EXACT);
		upper = 0.0;
	}

	double x0;
	double err = 0.0;
	int nterms = 10;
	while (upper - lower > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		res = solveOmega12(table, x1, x2, x0, &err, &nterms, size, lower_delta, upper_delta) - omega12;

		if (DEBUG)
			assert (res == res);

		if (res > 0.0)
			lower = x0;
		else
			upper = x0;
		iter++;
	}
	lambda = x0;

	printf("Lambda: %f\n", lambda);
	fflush(stdout);

	double distance;
	if (timelike) {
		idata.lower = tau[past_idx];
		idata.upper = tau[future_idx];
		distance = integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
	} else {
		idata.lower = tau[past_idx];
		idata.upper = geodesicMaxTau(manifold, lambda);
		distance = integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);

		idata.lower = tau[future_idx];
		distance += integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
	}

	gsl_integration_workspace_free(idata.workspace);

	printf("Distance: %f\n", distance);
	fflush(stdout);

	return distance;
}

//Returns the exact distance between two nodes in 4D
//Uses a pre-generated lookup table
//O(xxx) Efficiency (revise this)
inline double distance_v1(const double * const table, const float4 &node_a, const float tau_a, const float4 &node_b, const float tau_b, const int &dim, const Manifold &manifold, const double &a, const double &alpha, const long &size, const bool &compact)
{
	if (DEBUG) {
		assert (manifold == DE_SITTER || manifold == FLRW);
		if (manifold == FLRW) {
			assert (table != NULL);
			assert (alpha > 0.0);
			assert (size > 0);
			assert (!compact);
		}
		assert (dim == 3);
		assert (a > 0.0);
	}

	//Check if they are the same node
	if (node_a.w == node_b.w && node_a.x == node_b.x && node_a.y == node_b.y && node_a.z == node_b.z)
		return 0.0;

	bool DIST_DEBUG = true;

	IntData idata = IntData();
	idata.limit = 60;
	idata.tol = 1e-4;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	
	FastIntMethod method;
	double (*kernel)(double x, void *params);
	double distance;
	double lambda;
	double tau_max;

	if (manifold == FLRW) {
		lambda = lookupValue4D(table, size, (alpha / a) * SQRT(flatProduct_v2(node_a, node_b), STL), tau_a, tau_b);
		kernel = &flrwDistKernel;
		method = QAG;
		idata.key = GSL_INTEG_GAUSS61;
	} else if (manifold == DE_SITTER) {
		lambda = lookupValue4D(table, size, ACOS(sphProduct_v2(node_a, node_b), STL, VERY_HIGH_PRECISION), tau_a, tau_b);
		kernel = &deSitterDistKernel;
		method = QAG;
		idata.key = GSL_INTEG_GAUSS61;
	} else
		kernel = NULL;

	//Check for NaN
	if (lambda != lambda)
		return -1.0;

	if (DIST_DEBUG) {
		printf("\t\tLambda: %f\n", lambda);
		fflush(stdout);
	}

	tau_max = geodesicMaxTau(manifold, lambda);

	if (lambda == 0.0)
		distance = INF;
	else if (lambda > 0.0) {
		if (tau_a < tau_b) {
			idata.lower = tau_a;
			idata.upper = tau_b;
		} else {
			idata.lower = tau_b;
			idata.upper = tau_a;
		}
		distance = integrate1D(kernel, (void*)&lambda, &idata, method);
	} else if (lambda < 0 && (tau_a < tau_max && tau_b < tau_max)) {
		idata.lower = tau_a;
		idata.upper = tau_max;
		distance = integrate1D(kernel, (void*)&lambda, &idata, method);

		idata.lower = tau_b;
		distance += integrate1D(kernel, (void*)&lambda, &idata, method);
	} else
		distance = INF;

	gsl_integration_workspace_free(idata.workspace);

	if (DIST_DEBUG) {
		printf("\t\tDistance: %f\n", distance);
		fflush(stdout);
	}

	return distance;
}

//Returns the embedded distance between two nodes in 5D
//O(xxx) Efficiency (revise this)
inline double distanceEmb(const float4 &node_a, const float &tau_a, const float4 &node_b, const float &tau_b, const int &dim, const Manifold &manifold, const double &a, const double &alpha, const bool &compact)
{
	if (DEBUG) {
		assert (dim == 3);
		assert (manifold == DE_SITTER || manifold == FLRW);
		assert (compact);
	}

	//Check if they are the same node
	if (node_a.w == node_b.w && node_a.x == node_b.x && node_a.y == node_b.y && node_a.z == node_b.z)
		return 0.0;

	double alpha_tilde = alpha / a;
	double inner_product_ab;
	double distance;

	double z0_a = 0.0, z0_b = 0.0;
	double z1_a = 0.0, z1_b = 0.0;

	if (manifold == FLRW) {
		IntData idata = IntData();
		idata.tol = 1e-5;

		//Solve for z1 in Rotated Plane
		double power = 2.0 / 3.0;
		z1_a = alpha_tilde * POW(SINH(1.5 * tau_a, APPROX ? FAST : STL), power, APPROX ? FAST : STL);
		z1_b = alpha_tilde * POW(SINH(1.5 * tau_b, APPROX ? FAST : STL), power, APPROX ? FAST : STL);

		//Use Numerical Integration for z0
		idata.upper = z1_a;
		z0_a = integrate1D(&embeddedZ1, (void*)&alpha_tilde, &idata, QNG);
		idata.upper = z1_b;
		z1_b = integrate1D(&embeddedZ1, (void*)&alpha_tilde, &idata, QNG);
	} else if (manifold == DE_SITTER) {
		z0_a = SINH(tau_a, APPROX ? FAST : STL);
		z0_b = SINH(tau_b, APPROX ? FAST : STL);

		z1_a = COSH(tau_a, APPROX ? FAST : STL);
		z1_b = COSH(tau_b, APPROX ? FAST : STL);
	}

	if (DIST_V2)
		inner_product_ab = z1_a * z1_b * sphProduct_v2(node_a, node_b) - z0_a * z0_b;
	else
		inner_product_ab = z1_a * z1_b * sphProduct_v1(node_a, node_b) - z0_a * z0_b;

	if (manifold == FLRW)
		inner_product_ab /= POW2(alpha_tilde, EXACT);

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

//Returns the hyperbolic distance between two nodes in 2D
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
