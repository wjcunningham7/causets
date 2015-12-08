#ifndef GEODESICS_H_
#define GEODESICS_H_

#include "Causet.h"
#include "CuResources.h"
#include "Operations.h"

#include <boost/math/special_functions/gamma.hpp>

#define tgld(x) boost::math::tgamma<long double>(x)

/////////////////////////////
//(C) Will Cunningham 2015 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Variables used to reduce calculations

//k1[i] = 1.5*i
static const double k1[] = { 0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5, 30.0, 31.5, 33.0, 34.5, 36.0, 37.5, 39.0, 40.5, 42.0, 43.5, 45.0, 46.5, 48.0, 49.5, 51.0, 52.5, 54.0, 55.5, 57.0, 58.5, 60.0, 61.5, 63.0, 64.5, 66.0, 67.5, 69.0, 70.5, 72.0, 73.5, 75.0 };

//k2[i] = 6*i+1
static const double k2[] = { 1.0, 7.0, 13.0, 19.0, 25.0, 31.0, 37.0, 43.0, 49.0, 55.0, 61.0, 67.0, 73.0, 79.0, 85.0, 91.0, 97.0, 103.0, 109.0, 115.0, 121.0, 127.0, 133.0, 139.0, 145.0, 151.0, 157.0, 163.0, 169.0, 175.0, 181.0, 187.0, 193.0, 199.0, 205.0, 211.0, 217.0, 223.0, 229.0, 235.0, 241.0, 247.0, 253.0, 259.0, 265.0, 271.0, 277.0, 283.0, 289.0, 295.0, 301.0 };

//k3[i] = 6*i+2
static const double k3[] = { 2.0, 8.0, 14.0, 20.0, 26.0, 32.0, 38.0, 44.0, 50.0, 56.0, 62.0, 68.0, 74.0, 80.0, 86.0, 92.0, 98.0, 104.0, 110.0, 116.0, 122.0, 128.0, 134.0, 140.0, 146.0, 152.0, 158.0, 164.0, 170.0, 176.0, 182.0, 188.0, 194.0, 200.0, 206.0, 212.0, 218.0, 224.0, 230.0, 236.0, 242.0, 248.0, 254.0, 260.0, 266.0, 272.0, 278.0, 284.0, 290.0, 296.0, 302.0 };

//================================================//
// Approximtions to omega = f(tau1, tau2, lambda) //
//================================================//

//----------//
// Region 1 //
//----------//

inline double omegaRegion1(const double &x, const double &lambda, const double &z, double * const err, int * const nterms)
{
	#if DEBUG
	assert (err != NULL);
	assert (nterms != NULL);
	assert (x >= 0.0 && x <= GEODESIC_LOWER);
	assert (lambda != 0.0);
	assert (z != 0.0);
	assert (*err >= 0.0);
	assert (!(*err == 0.0 && *nterms == -1));
	#endif

	double omega = 0.0;

	//Used for _2F1
	HyperType ht;
	double f;
	double er, f_err;
	int nt, f_nt;

	//Determine which transformation of 2F1 is used
	//printf("z: %f\n", z);
	ht = getHyperType(z);
	//printf("w: %f\n", ht.w);
	er = 1.0E-8;
	nt = getNumTerms(ht.w, er);
	//printf("NTerms: %d\n", nt);

	double buf = 0.0;
	switch (ht.type) {
	case 1:
		buf = sqrt(ht.w);
		break;
	case 3:
		buf = 1.0 / ht.w;
		break;
	default:
		break;
	}

	//Series solution (see notes)
	double error = INF;
	double omega_k;
	int k = 0;

	while (error > *err && k < *nterms) {
		f = 0.0;
		f_err = er;
		f_nt = nt;

		omega_k = 0.0;
		switch (ht.type) {
		case 0:
			// 0 <= z <= 0.5
			//printf("CASE 0.\n");
			_2F1_F(0.5, k1[k] + 0.25, k1[k] + 1.25, ht.w, &f, &f_err, &f_nt);
			omega_k = f;
			break;
		case 1:
			// 0.5 < z <= 1
			//printf("CASE 1.\n");
			omega_k = SQRT_PI * tgamma(k1[k] + 1.25) * pow(z, -1.0 * (k1[k] + 0.25)) / tgamma(k1[k] + 0.75);
			_2F1_F(k1[k] + 0.75, 1.0, 1.5, ht.w, &f, &f_err, &f_nt);
			omega_k -= k2[k] * buf * f / 2.0;
			break;
		case 2:
			// -1 <= z < 0
			//printf("CASE 2.\n");
			_2F1_F(0.5, 1.0, k1[k] + 1.25, ht.w, &f, &f_err, &f_nt);
			omega_k = f;
			break;
		case 3:
			// z < -1
			//printf("CASE 3.\n");
			omega_k = tgamma(k1[k] + 1.25) * tgamma(0.25 - k1[k]) * pow(buf - 1.0, -1.0 * (k1[k] + 0.25)) / SQRT_PI;
			f_nt = -1;
			_2F1(0.5, 1.0, 1.25 - k1[k], ht.w, &f, &f_err, &f_nt, false);
			omega_k += k2[k] * sqrt(ht.w) * f / (k2[k] - 2.0);
			break;
		default:
			// This should never be reached
			return NAN;
		}
		//printf("k: %d\tnt: %d\n", k, f_nt);

		omega_k *= pow(x, k2[k]);
		omega_k /= tgamma(k + 1.0) * tgamma(0.5 - k) * k2[k];
		//printf("k: %d\tf: %.8e\tomega_k: %.8e\n", k, f, omega_k);

		omega += omega_k;
		error = fabs(omega_k / omega);
		k++;
	}

	if (ht.type == 2)
		omega *= sqrt(1.0 - ht.w);

	omega *= 2.0 * SQRT_PI;

	//*nterms = k;
	//*err = error;

	return omega;
}

//----------//
// Region 2 //
//----------//

//Supplementary functions used for elliptic integral approximations
//Described in [4]
inline double theta_nm(const int &n, const int &m)
{
	#if DEBUG
	assert (n >= 0);
	assert (m >= 1);
	#endif

	return static_cast<double>(m * M_PI / (2.0 * n + 1.0));
}

inline double sigma_nm(const int &n, const int &m)
{
	#if DEBUG
	assert (n >= 0);
	assert (m >= 1);
	#endif

	return SQRT(1.0 + POW2(SIN(theta_nm(n, m), APPROX ? FAST : STL), EXACT), STL);
}

inline double rho_nm(const int &n, const int &m)
{
	#if DEBUG
	assert (n >= 0);
	assert (m >= 1);
	#endif

	return SQRT(1.0 + POW2(COS(theta_nm(n, m), APPROX ? FAST : STL), EXACT), STL);
}

//Returns a complex phi for lambda > 0
inline double2 ellipticPhi(const double &x, const double &lambda)
{
	#if DEBUG
	assert (x > 0.0);
	assert (lambda > 0.0);
	#endif

	double2 phi;
	double z = lambda * POW2(POW2(x, EXACT), EXACT);
	double sqz = SQRT(z, STL);

	double a = sqz / 2.0;
	double t1 = SQRT(1.0 + 0.25 / POW2(a, EXACT), STL);
	double t2 = 0.5 / a;
	
	phi.x = -1.0 * ATAN(SQRT(t1 - t2, STL), STL, VERY_HIGH_PRECISION);
	//printf("phiR: %f\n", phi.x);

	#if DEBUG
	assert (phi.x > -1.0 * HALF_PI / 2.0);
	#endif

	double t3 = 2.0 * a;
	double t4 = t3 + SQRT(POW2(t3, EXACT) + 1.0, STL);
	double t5 = SQRT(POW2(t4, EXACT) - 1.0, STL);

	phi.y = 0.5 * LOG(t4 + t5, APPROX ? FAST : STL);
	//printf("phiI: %f\n", phi.y);

	#if DEBUG
	assert (phi.y >= 0.0);
	#endif

	return phi;
}

//xi(phi, tau_m) = arctan(tau_m * tan(phi))
inline double2 ellipticXi(const double2 &upsilon, const double &tau_nm)
{
	#if DEBUG
	assert (upsilon.x <= 0.0);
	assert (upsilon.y >= 0.0 && upsilon.y < 1.0);
	assert (tau_nm > 0.0);
	#endif

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

	#if DEBUG
	assert (xi.x < 0.0);
	assert (xi.y > 0.0);
	#endif

	return xi;
}

//Approximates the incomplete elliptic integral of the first kind with k^2 = -1
//This form assumes phi is real-valued
inline double ellipticIntF(const double &phi, const int &n)
{
	#if DEBUG
	//assert (phi % HALF_PI != 0.0);
	assert (n >= 0);
	#endif

	double f = 0.0;
	double s;
	int m;

	for (m = 1; m <= n; m++) {
		s = sigma_nm(n, m);
		f += ATAN(s * TAN(phi, APPROX ? FAST : STL), STL, VERY_HIGH_PRECISION) / s;
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
	#if DEBUG
	assert (phi.x > -0.25 * M_PI && phi.x <= 0.0);
	assert (phi.y >= 0.0);
	assert (n >= 0);
	#endif

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
	#if DEBUG
	//assert (phi % HALF_PI != 0.0);
	assert (n >= 0);
	#endif

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
	#if DEBUG
	assert (phi.x > -0.25 * M_PI && phi.x <= 0.0);
	assert (phi.y >= 0.0);
	assert (n >= 0);
	#endif

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
	#if DEBUG
	assert (err != NULL);
	assert (nterms != NULL);
	assert (x >= GEODESIC_LOWER && x <= GEODESIC_UPPER);
	assert (lambda > 0.0);
	assert (*err >= 0.0);
	assert (!(*err == 0.0 && *nterms == -1));
	#endif

	double2 phi = ellipticPhi(x, lambda);
	double elF_plus = ellipticIntF_Complex(phi, *nterms, true);
	double elF_minus = ellipticIntF_Complex(phi, *nterms, false);
	double elE_minus = ellipticIntE_Complex(phi, *nterms, false);
	double sql = SQRT(lambda, STL);

	//printf("elF_plus:\t%.8e\n", elF_plus);
	//printf("elF_minus:\t%.8e\n", elF_minus);
	//printf("elE_minus:\t%.8e\n", elE_minus);

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
	#if DEBUG
	assert (err != NULL);
	assert (nterms != NULL);
	assert (x >= GEODESIC_LOWER && x <= GEODESIC_UPPER);
	assert (lambda < 0.0);
	assert (*err >= 0.0);
	assert (!(*err == 0.0 && *nterms == -1));
	#endif

	double s = POW(-1.0 * lambda, 0.25, APPROX ? FAST : STL) * x;
	if (ABS(s - 1.0, STL) < 1.0E-14)
		s = 1.0;
	double t = 1.0 - POW2(POW2(s, EXACT), EXACT);
	double u = POW2(s, EXACT);
	double sql = SQRT(-1.0 * lambda, STL);

	double phi = asin(s);
	//printf("phi: %.16e\n", phi);
	double el_F = ellipticIntF(phi, *nterms);
	double el_E = ellipticIntE(phi, *nterms);
	//printf("F: %f\tE: %f\n", el_F, el_E);

	double omega = (55.0 * SQRT(t, STL) - 153.0 * sql * asin(u)) / (16.0 * SQRT(2.0, STL) * lambda);
	omega -= 21.0 * el_F / (8.0 * SQRT(2.0 * sql, STL));
	omega += 171.0 * (el_F - el_E) / (8.0 * SQRT(2.0 * POW3(sql, EXACT), STL));

	return omega;
}

//----------//
// Region 3 //
//----------//

//Supplementary functions for Region 3
//Extreme precision is needed here
//See notes for how/why they are used

inline long double ln_hnl(const double &x, const double &lambda, const int &l, const int &n)
{
	#if DEBUG
	assert (x >= GEODESIC_UPPER);
	assert (lambda != 0.0);
	assert (l >= 0 && !(l % 2));
	assert (n >= 0 && n <= (3 * l) / 2);
	#endif

	long double t1 = - 1.5L * l - 0.5L;
	long double t2 = n + t1;
	long double t3 = 1.0L + static_cast<long double>(lambda * x * x * x * x);

	return log(tgld(t2) * sin(t2 * M_PI)) - log(tgld(t1) * sin(t1 * M_PI)) - log(tgld(n + 1.0L)) - n * log(t3);
}

inline long double ln_fl(const double &x, const double &lambda, const int &l)
{
	#if DEBUG
	assert (x >= GEODESIC_UPPER);
	assert (lambda != 0.0);
	assert (l >= 0 && !(l % 2));
	#endif

	long double t1 = log(tgld(1.5L * l + 1.000001L));
	long double t2 = -1.5L * l - 0.5L;
	long double sum = 0.0L;
	long double t3;

	for (int n = 1; n <= 3 * l / 2; n++) {
		t3 = n + t2;
		//long double lnhnl = ln_hnl(x, lambda, l, n);
		//printf("n: %d\tlnhnl: %Lf\n", n, lnhnl);
		sum += exp(ln_hnl(x, lambda, l, n)) / sin(t3 * M_PI);
	}
	sum *= sin(t2 * M_PI);
	//printf("sum: %Lf\n", sum);

	return t1 + log(1.0L + sum);
}

//Must multiply by csc((1-3l)pi/2) after taking e^(ln_Fl)
inline long double ln_Fl(const double &x, const double &lambda, const int &l)
{
	#if DEBUG
	assert (x >= GEODESIC_UPPER);
	assert (lambda != 0.0);
	assert (l >= 0 && !(l % 2));
	#endif

	long double t1 = -1.0L * log(SQRT_PI);
	long double t2 = -1.5L * l + 0.5L;
	long double t3 = 1.0L + static_cast<long double>(lambda * x * x * x * x);

	long double t4 = log(tgld(t2) * sin(t2 * M_PI));
	long double t5 = (1.0L - t2) * log(t3);

	return t1 + t4 + t5 + ln_fl(x, lambda, l);
}

//Calculates the even terms in the series for x, lambda
inline double omegaRegion3a(const double &x, const double &lambda, const double &z, double * const err, int * const nterms)
{
	#if DEBUG
	assert (err != NULL);
	assert (nterms != NULL);
	assert (x >= GEODESIC_UPPER);
	assert (lambda != 0.0);
	assert (z != 0.0);
	assert (*err >= 0.0);
	assert (!(*err == 0.0 && *nterms == -1));
	#endif

	double omega3a = 0.0;

	//Used for _2F1
	HyperType ht;
	double f;
	double er, f_err;
	int nt, f_nt;

	//Determine which transformation of 2F1 is used for the even terms
	//printf("z: %.16e\n", z);
	ht = getHyperType(z);
	//printf("w: %.16e\n", ht.w);
	er = 1.0E-6;
	nt = getNumTerms(ht.w, er);

	double buf;
	switch (ht.type) {
	case 1:
		buf = sqrt(ht.w);
		break;
	case 2:
		buf = 1.0 / sqrt(1.0 - z);
		break;
	case 3:
		buf = 0.000001;
		break;
	default:
		buf = NAN;
		break;
	}

	//Calculate the even terms in the series
	double error_l = INF;
	double omega_l;
	int l = 0;
	int p;

	while (error_l > *err && l < *nterms) {
		f = 0.0;
		f_err = er;
		f_nt = nt;

		omega_l = 0.0;
		switch (ht.type) {
		case 0:
			// 0 <= z <= 0.5
			//printf("CASE 0.\n");
			//_2F1(0.5, -1.0 * k1[l] - 0.5, 0.5 - k1[l], ht.w, &f, &f_err, &f_nt, false);
			_2F1_F(0.5, -1.0 * k1[l] - 0.5, 0.5 - k1[l], ht.w, &f, &f_err, &f_nt);
			omega_l = f;
			break;
		case 1:
			// 0.5 < z <= 1
			//printf("CASE 1.\n");
			f_nt = (l > 0 && k1[l] + 1.0 < f_nt) ? static_cast<int>(k1[l] + 1.0) : f_nt;
			//_2F1(-1.0 * k1[l], 1.0, 1.5, ht.w, &f, &f_err, &f_nt, false);
			_2F1_F(-1.0 * k1[l], 1.0, 1.5, ht.w, &f, &f_err, &f_nt);
			omega_l = k3[l] * buf * f / 2.0;
			break;
		case 2:
			// -1 <= z < 0
			//printf("CASE 2.\n");
			//_2F1(0.5, 1.0, 0.5 - k1[l], ht.w, &f, &f_err, &f_nt, false);
			_2F1_F(0.5, 1.0, 0.5 - k1[l], ht.w, &f, &f_err, &f_nt);
			omega_l = buf * f;
			break;
		case 3:
			// z < -1
			//printf("CASE 3.\n");
			for (p = 0; p <= k1[l]; p++)
				omega_l += pow(-1.0, p) * POCHHAMMER(-1.0 * k1[l] - 0.5, p) * POCHHAMMER(-1.0 * k1[l], p) * tgamma(k1[l] + 1.0 - p + buf) * pow(ht.w, p - k1[l] - 0.5) / tgamma(p + 1.0);
			omega_l *= tgamma(0.5 - k1[l]) / SQRT_PI;
			f = omega_l;	//
			break;
		default:
			// This should never be reached
			return NAN;
		}

		omega_l /= -1.0 * k3[l] * pow(x, k3[l]) * tgamma(l + 1.0) * tgamma(0.5 - l);
		//printf("l: %d\tf: %.8e\tomega_l: %.8e\n", l, f, omega_l);

		omega3a += omega_l;
		error_l = fabs(omega_l / omega3a);
		l += 2;
	}

	omega3a *= 2.0 * SQRT_PI;

	if (!!omega3a)
		*nterms = l - 1;
	*err = error_l;

	return omega3a;
}

//Calculates the even terms in the series for x1, x2, lambda
//Note this is used when lambda is large and the Hypergeometric function diverges
inline double omegaRegion3aDiv(const double &x1, const double &x2, const double &lambda, double * const err, int * const nterms)
{
	#if DEBUG
	assert (err != NULL);
	assert (nterms != NULL);
	assert (x1 >= GEODESIC_UPPER);
	assert (x2 >= GEODESIC_UPPER);
	assert (x2 > x1);
	assert (lambda != 0.0);
	assert (*err >= 0.0);
	assert (!(*err == 0.0 && *nterms == -1));
	#endif

	double omega3aD = 0.0;

	//Calculate the even terms in the series
	double error_l = INF;
	double omega_l, csck1;

	long double z12, Z12;
	long double lnF1, lnF2;

	int l = 0;

	while (error_l > *err && l < *nterms) {
		omega_l = 0.0;
		csck1 = 1.0 / sin((0.5 - k1[l]) * M_PI);

		//Calculate omega_l for diverging case
		lnF1 = ln_Fl(x1, lambda, l);
		lnF2 = ln_Fl(x2, lambda, l);
		z12 = -1.0L * static_cast<long double>(k3[l] * log(x1 / x2)) + lnF1 - lnF2;

		if (fabs(z12) < 0.001L)
			//Taylor Expansion for 1 - e^z12
			Z12 = -1.0L * z12 * (1.0L + 0.5L * z12 * (1.0L + z12 * (1.0L + 0.25L * z12) / 3.0L));
		else
			Z12 = 1.0L - exp(z12);

		if (fabs(Z12) > 1.0e-14L) {
			omega_l = csck1 * pow(x2, -1.0 * k3[l]) * static_cast<double>(exp(lnF2) * Z12) / (-1.0 * k3[l] * tgamma(l + 1.0) * tgamma(0.5 - l));
			//printf("l: %d\tF1: %Lf\tF2: %Lf\tomega_l: %e\n", l, exp(lnF1) * static_cast<long double>(csck1), exp(lnF2) * static_cast<long double>(csck1), omega_l);
			//printf("l: %d\tZ12: %Le\tomega_l: %e\n", l, Z12, omega_l);
			omega3aD += omega_l;
			error_l = fabs(omega_l / omega3aD);
			l += 2;
		} else {
			l++;
			break;
		}
	}

	omega3aD *= 2.0 * SQRT_PI;

	*nterms = l - 1;
	*err = error_l;

	return omega3aD;
}

//Calculates the odd terms in the series for x, lambda
//z should be sqrt(1 + lambda x^4)
inline double omegaRegion3b(const double * const table, const double &x, const double &lambda, const double &z, double * const err, int * const nterms, const long &size)
{
	#if DEBUG
	assert (table != NULL);
	assert (err != NULL);
	assert (nterms != NULL);
	assert (x >= GEODESIC_UPPER);
	assert (lambda != 0.0);
	assert (*err >= 0.0);
	assert (!(*err == 0.0 && *nterms == -1));
	assert (size > 0L);
	assert (*nterms <= 100);	//Limit of the lookup table
	#endif

	double omega3b = 0.0;
	int p;

	//Calculate the odd terms in the series
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
		n1 = 0.5 * pow(lambda, n3 - 1.0);
		start = n3 * (n - 1);

		omega_m += table[start] * log(z1);
		omega_m += table[start+n3] * log(fabs(z2));
		//printf("A1 = %f\tB1 = %f\n", table[start], table[start+n3]);
		for (p = 1; p < n3; p++) {
			omega_m += table[start+p] / (-1.0 * p * pow(z1, p));
			omega_m += table[start+n3+p] / (-1.0 * p * pow(z2, p));
			//printf("A%d = %f\tB%d = %f\n", p+1, table[start+p], p+1, table[start+n3+p]);
		}

		omega_m *= n1;
		omega_m /= tgamma(m + 1.0) * tgamma(0.5 - m);
		//printf("m: %d\tomega_m: %.8e\n", m, omega_m);

		omega3b += omega_m;
		error_m = fabs(omega_m / omega3b);
		m += 2;
	}

	omega3b *= 2.0 * SQRT_PI;

	if (!!omega3b)
		*nterms = m - 1;
	*err = error_m;

	return omega3b;
}

//Region 3
inline double omegaRegion3(const double * const table, const double &x1, const double &x2, const double &lambda, const double &z1, const double &z2, double * const err, int * const nterms, const long &size)
{
	#if DEBUG
	assert (table != NULL);
	assert (err != NULL);
	assert (nterms != NULL);
	assert (x1 >= GEODESIC_UPPER);
	assert (x2 >= GEODESIC_UPPER);
	assert (x2 > x1);
	assert (lambda != 0.0);
	assert (z1 != 0.0);
	assert (z2 != 0.0);
	assert (*err > 0.0);	
	assert (!(*err == 0.0 && *nterms == -1));
	assert (size > 0L);
	assert (*nterms <= 100);	//Limit of the lookup table
	#endif

	double omega;
	double er, final_err;
	int nt, final_nterms;

	er = *err;
	nt = *nterms;
	//This value is used because when lambda is larger the 2F1 term begins to diverge
	if (ABS(lambda, STL) < 0.9) {
		omega = omegaRegion3a(x2, lambda, z2, err, nterms);
		final_err = *err;

		*err = er;
		*nterms = nt;
		omega -= omegaRegion3a(x1, lambda, z1, err, nterms);
		final_err = final_err > *err ? final_err : *err;
		final_nterms = *nterms;
		*nterms = nt;
	} else {
		omega = omegaRegion3aDiv(x1, x2, lambda, err, nterms);
		final_err = *err;
		final_nterms = *nterms;
	}

	double za1 = sqrt(1.0 - z1);
	double za2 = sqrt(1.0 - z2);

	*err = er;
	omega += omegaRegion3b(table, x2, lambda, za2, err, nterms, size);
	final_err = final_err > *err ? final_err : *err;

	*err = er;
	omega -= omegaRegion3b(table, x1, lambda, za1, err, nterms, size);
	final_err = final_err > *err ? final_err : *err;
	final_nterms = final_nterms > *nterms ? final_nterms : *nterms;

	//*err = final_err;
	//*nterms = final_nterms;

	return omega;
}

//Gives omega = f(x2, lambda) - f(x1, lambda) using numerical approximations
//This function is an intermediary subroutine designed to pick the correct regions for x1 and x2
//NOTE: x is related to tau by x = sinh(1.5*tau)^(1/3)
inline double omegaRegionXY(const double * const table, const double &x1, const double &x2, const double &lambda, const double &z1, const double &z2, double * const err, int * const nterms, const long &size)
{
	#if DEBUG
	assert (table != NULL);
	assert (err != NULL);
	assert (nterms != NULL);
	assert (x1 >= 0.0);
	assert (x2 >= 0.0);
	assert (x2 > x1);
	assert (lambda != 0.0);
	assert (z1 != 0.0);
	assert (z2 != 0.0);
	assert (*err >= 0.0);
	assert (!(*err == 0.0 && *nterms == -1));
	assert (size > 0L);
	assert (*nterms <= 100);	//Limit of the lookup table for Region 3
	#endif

	double omega;
	double z_lower = -1.0 * lambda * POW2(POW2(GEODESIC_LOWER, EXACT), EXACT);
	double z_upper = -1.0 * lambda * POW2(POW2(GEODESIC_UPPER, EXACT), EXACT);

	if (x2 < GEODESIC_LOWER)
		//Both in Region 1
		omega = omegaRegion1(x2, lambda, z2, err, nterms) - omegaRegion1(x1, lambda, z1, err, nterms);
	else if (x1 >= GEODESIC_UPPER)
		//Both in Region 3
		omega = omegaRegion3(table, x1, x2, lambda, z1, z2, err, nterms, size);
	else if (x1 >= GEODESIC_LOWER && x2 < GEODESIC_UPPER)
		//Both in Region 2
		omega = omegaRegion2a(x2, lambda, err, nterms) - omegaRegion2a(x1, lambda, err, nterms);
	else {
		if (x2 < GEODESIC_UPPER)
			//x1 in Region 1, x2 in Region 2
			omega = omegaRegion2a(x2, lambda, err, nterms) - omegaRegion2a(GEODESIC_LOWER, lambda, err, nterms) + omegaRegion1(GEODESIC_LOWER, lambda, z_lower, err, nterms) - omegaRegion1(x1, lambda, z1, err, nterms);
		else if (x1 >= GEODESIC_LOWER)
			//x1 in Region 2, x2 in Region 3
			omega = omegaRegion3(table, GEODESIC_UPPER, x2, lambda, z_upper, z2, err, nterms, size) + omegaRegion2a(GEODESIC_UPPER, lambda, err, nterms) - omegaRegion2a(x1, lambda, err, nterms);
		else
			//x1 in Region 1, x2 in Region 3
			omega = omegaRegion3(table, GEODESIC_UPPER, x2, lambda, z_upper, z2, err, nterms, size) + omegaRegion2a(GEODESIC_UPPER, lambda, err, nterms) - omegaRegion2a(GEODESIC_LOWER, lambda, err, nterms) + omegaRegion1(GEODESIC_LOWER, lambda, z_lower, err, nterms) - omegaRegion1(x1, lambda, z1, err, nterms);				
	}

	return omega;
}

//Maximum Time in Geodesic (non-embedded)
//Returns tau_max=f(lambda) with lambda < 0
inline double geodesicMaxTau(const Manifold &manifold, const double &lambda)
{
	#if DEBUG
	assert (manifold == DE_SITTER || manifold == DUST || manifold == FLRW);
	assert (lambda < 0.0);
	#endif

	if (manifold == FLRW)
		return (2.0 / 3.0) * ASINH(POW(-lambda, -0.75, STL), STL, VERY_HIGH_PRECISION);
	else if (manifold == DUST)
		return (2.0 / 3.0) * POW(-lambda, -0.75, STL);
	else if (manifold == DE_SITTER) {
		double g = POW(-lambda, -0.5, STL);
		return g >= 1.0 ? ACOSH(g, STL, VERY_HIGH_PRECISION) : 0.0;
	}

	return 0.0;
}

//Maximum X Time in Dust or FLRW Geodesic (non-embedded)
//Returns x_max = f(lambda) with lambda < 0
inline double geodesicMaxX(const double &lambda)
{
	#if DEBUG
	assert (lambda < 0.0);
	#endif

	return POW(-1.0 * lambda, -0.25, APPROX ? FAST : STL);
}

//Gives omega12 = f(x1, x2, lambda) using numerical approximations
//This function should be faster than the numerical integral defined in the kernel
//functions, flrwLookupKernel or flrwLookupKernelX
//NOTE: x is related to tau by x = sinh(1.5*tau)^(1/3)
inline double solveOmega12(const double * const table, const double &x1, const double &x2, const double &lambda, const double &z1, const double &z2, double * const err, int * const nterms, const long &size)
{
	#if DEBUG
	assert (table != NULL);
	assert (err != NULL);
	assert (nterms != NULL);
	assert (x1 >= 0.0);
	assert (x2 >= 0.0);
	assert (lambda != 0.0);
	assert (z1 != 0.0);
	assert (z2 != 0.0);
	assert (*err >= 0.0);
	assert (!(*err == 0.0 && *nterms == -1));
	assert (size > 0L);
	assert (*nterms <= 100);	//Limit of the lookup table for Region 3
	#endif

	double omega = 0.0;
	double zm = 1.0;
	double x_max;

	if (lambda < 0.0) {
		//Spacelike
		x_max = geodesicMaxX(lambda);
		if (x1 > x_max || x2 > x_max)
			return NAN;

		omega = omegaRegionXY(table, x1, x_max, lambda, z1, zm, err, nterms, size) + omegaRegionXY(table, x2, x_max, lambda, z2, zm, err, nterms, size);
	} else
		//Timelike
		omega = omegaRegionXY(table, x1, x2, lambda, z1, z2, err, nterms, size);

	return omega;
}

//Embedded Z1 Coordinate used in Naive Embedding
//For use with GNU Scientific Library
inline double embeddedZ1(double x, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (x >= 0.0);
	#endif

	double alpha_tilde = ((double*)params)[0];

	#if DEBUG
	assert (alpha_tilde != 0.0);
	#endif

	return SQRT(1.0 + (x / (POW3(alpha_tilde, EXACT) + POW3(x, EXACT))), STL);
}

//Integrands for Exact Geodesic Calculations
//For use with GNU Scientific Library

//Distance Kernels

inline double deSitterDistKernel(double x, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (x >= 0.0);
	#endif

	double *p = (double*)params;
	double lambda = p[0];

	double lcx2 = lambda * POW2(COSH(x, STL), EXACT);
	double distance = SQRT(ABS(lcx2 / (1.0 + lcx2), STL), STL);

	return distance;
}

inline double dustDistKernel(double x, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (x >= 0.0);
	#endif

	double lambda = ((double*)params)[0];

	double g43 = lambda * POW(1.5 * x, 4.0 / 3.0, STL);
	double distance = SQRT(ABS(g43 / (1.0 + g43), STL), STL);

	return distance;
}

inline double flrwDistKernel(double x, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (x >= 0.0);
	#endif

	double lambda = ((double*)params)[0];

	double sx = sinh(1.5 * x);
	double lsx43 = lambda * pow(sx, 4.0 / 3.0);
	double distance = sqrt(fabs(lsx43 / (1.0 + lsx43)));

	return distance;
}

//Transcendental Kernels Solving omega12=f(tau1,tau2,lambda)
//These do not use the approximations - they use the exact form of the integrals

inline double flrwLookupKernel(double x, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (x >= 0);
	#endif

	double lambda = ((double*)params)[0];

	double sx = sinh(1.5 * x);
	double sx43 = pow(sx, 4.0 / 3.0);
	double g = sx43 + lambda * sx43 * sx43;
	double omega12 = 1.0 / sqrt(g);

	return omega12;
}

//Same as flrwLookupKernel but uses a change of variables
//x = sinh(1.5*tau)^(1/3)
//to make calculations faster
//Multiply by 2 afterwards
inline double flrwLookupKernelX(double x, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (x >= 0.0);
	#endif

	double lambda = ((double*)params)[0];

	double x4 = x * x * x * x;
	double x6 = x4 * x * x;

	double g = 1.0 + lambda * x4;
	double omega12 = 1.0 / sqrt(g * (1.0 + x6));

	return omega12;
}

//x = (1.5*tau)^(4/3)
//Multiply by 2 afterwards
inline double dustLookupKernel(double x, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (x >= 0.0);
	#endif

	double lambda = ((double*)params)[0];

	double x4 = POW2(POW2(x, EXACT), EXACT);
	double omega12 = 1.0 / SQRT(1.0 + lambda * x4, STL);

	return omega12;
}

inline double deSitterLookupKernel(double x, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (x >= 0);
	#endif

	double lambda = ((double*)params)[0];

	double cx2 = cosh(x) * cosh(x);
	double g = cx2 + lambda * cx2 * cx2;
	double omega12 = 1.0 / sqrt(g);

	return omega12;
}

//The exact solution in de Sitter (the integral can be solved analytically)
inline double deSitterLookupExact(const double &tau, const double &lambda)
{
	double x = 0.0;
	//double g = 0.0;

	if (tau > LOG(MTAU, STL) / 6.0)
		x = exp(2.0 * tau) / 2.0;
	else
		x = COSH(2.0 * tau, STL);
	x += 1.0;
	x *= lambda;

	/*if (x > -2.0 && x < 0.0) {
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
	g += LOG(2.0, STL) / 2.0;*/

	x += 2.0;
	return atan(sqrt(2.0) * sinh(tau) / sqrt(x));
	
	//return ATAN(exp(g), STL, VERY_HIGH_PRECISION);
}

//Approximation for flrwLookupKernelX
//Assumes lambda is large so (1+lambda*x^4) => lambda*x^4
//Multiply by 2*lambda afterwards
//See how this is called by looking in distance_v2() below
inline double flrwLookupApprox(double x, void *params)
{
	#if DEBUG
	assert (x >= 0.0);
	#endif

	double x4 = POW2(POW2(x, EXACT), EXACT);
	double x6 = x4 * POW2(x, EXACT);

	return 1.0 / SQRT((x4 * (1.0 + x6)), STL);
}

//=====================//
// Distance Algorithms //
//=====================//

//Returns the distance between two nodes in the non-compact K = 0 FLRW manifold
//Version 2 does not use the lookup table for lambda = f(omega12, tau1, tau2)
//O(xxx) Efficiency (revise this)
inline double distanceFLRW(const double * const table, Coordinates *c, const float * const tau, const int &N_tar, const int &stdim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const long &size, const bool &symmetric, const bool &compact, int past_idx, int future_idx)
{
	#if DEBUG
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
	assert (stdim == 4);
	assert (manifold == FLRW);
	assert (a > 0.0);
	assert (zeta < HALF_PI);
	assert (r_max > 0.0);
	assert (alpha > 0.0);
	assert (size > 0L);
	assert (!compact);
	assert (past_idx >= 0 && past_idx < N_tar);
	assert (future_idx >= 0 && future_idx < N_tar);
	//assert (past_idx < future_idx);
	#endif

	if (future_idx < past_idx) {
		//Bitwise swap
		past_idx ^= future_idx;
		future_idx ^= past_idx;
		past_idx ^= future_idx;
	}

	IntData idata = IntData();
	idata.limit = 60;
	idata.tol = 1.0e-5;

	double x1 = pow(sinh(1.5 * tau[past_idx]), 1.0 / 3.0);
	double x2 = pow(sinh(1.5 * tau[future_idx]), 1.0 / 3.0);
	//printf("x1: %.16ee\tx2: %.16e\n", x1, x2);
	double omega12;
	double lambda;

	bool timelike = nodesAreRelated(c, N_tar, stdim, manifold, a, zeta, zeta1, r_max, alpha, symmetric, compact, past_idx, future_idx, &omega12);
	//printf("dt: %.16e\tdx: %f\n", c->w(future_idx) - c->w(past_idx), omega12);
	omega12 *= alpha / a;

	//Bisection Method
	double res = 1.0, tol = 1.0E-5;
	double lower, upper;
	int iter = 0, max_iter = 10000;

	if (!timelike) {
		lower = -1.0 / (x2 * x2 * x2 * x2);
		upper = 0.0;
		//printf("Spacelike.\n");
	} else {
		lower = 0.0;
		upper = 1000.0;
		//printf("Timelike.\n");
	}

	//Check if distance is infinite
	double inf_dist = 4.0 * tgamma(1.0 / 3.0) * tgamma(7.0 / 6.0) / sqrt(M_PI) - tauToEtaFLRWExact(tau[past_idx], 1.0, 1.0) - tauToEtaFLRWExact(tau[future_idx], 1.0, 1.0);
	if (omega12 > inf_dist)
		return INF;

	//printf("inf_dist: %f\n", inf_dist);

	//Use upper or lower branch of the solution
	double ml = -1.0 / (x2 * x2 * x2 * x2);
	bool upper_branch = false;

	idata.lower = x1;
	idata.upper = x2;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	//printf("x2 - x1 = %.8e\n", x2 - x1);
	double branch_cutoff;
	if (c->w(future_idx) - c->w(past_idx) > 1.0E-14) {
		branch_cutoff = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&ml, &idata, QAGS);
		if (branch_cutoff != branch_cutoff) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	} else
		branch_cutoff = 0.0;
	//printf("Branch cutoff: %f\n", branch_cutoff);
	if (!!branch_cutoff && fabs(omega12 - branch_cutoff) / branch_cutoff < 1.0e-3)
		omega12 = branch_cutoff;
	if (!timelike && omega12 > branch_cutoff)
		upper_branch = true;

	double x0, mx;
	while (fabs(res) > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		if (!timelike && upper_branch) {
			mx = geodesicMaxX(x0);
			//printf("[0] lambda: %f\tx1: %f\tx2: %f\txm: %f\n", x0, x1, x2, mx);
			idata.lower = x1;
			idata.upper = mx;
			res = integrate1D(&flrwLookupKernelX, (void*)&x0, &idata, QAGS);
			//assert (res == res);
			if (res != res) {
				gsl_integration_workspace_free(idata.workspace);
				return INF;
			}
			idata.lower = x2;
			res += integrate1D(&flrwLookupKernelX, (void*)&x0, &idata, QAGS);
			//assert (res == res);
			if (res != res) {
				gsl_integration_workspace_free(idata.workspace);
				return INF;
			}
			res *= 2.0;
			//printf("\tres: %e\n", res);
			res -= omega12;
			if (res < 0.0)
				lower = x0;
			else
				upper = x0;
		} else {
			//printf("[1] lambda: %f\tx1: %f\tx2: %f\n", x0, x1, x2);
			idata.lower = x1;
			idata.upper = x2;
			res = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&x0, &idata, QAGS);
			//assert (res == res);
			if (res != res) {
				gsl_integration_workspace_free(idata.workspace);
				return INF;
			}
			res -= omega12;
			if (res > 0.0)
				lower = x0;
			else
				upper = x0;
		}
		iter++;
	}
	lambda = x0;
	//printf("x1: %f\tx2: %f\tlam: %f\tdx: %f\n", x1, x2, lambda, omega12);

	//printf("Lambda: %f\n", lambda);
	//fflush(stdout);

	double distance;
	if (!timelike && upper_branch) {
		idata.lower = tau[past_idx];
		idata.upper = geodesicMaxTau(manifold, lambda);
		distance = integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
		//assert (distance == distance);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}

		idata.lower = tau[future_idx];
		distance += integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
		//assert (distance == distance);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	} else {
		idata.lower = tau[past_idx];
		idata.upper = tau[future_idx];
		distance = integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
		//assert (distance == distance);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	}

	gsl_integration_workspace_free(idata.workspace);

	//if (distance != distance)
	//	return INF;

	//printf("Distance: %f\n", distance);
	//fflush(stdout);

	return distance;
}

//Returns the geodesic distance for two points on a K = 0 dust manifold
inline double distanceDust(Coordinates *c, const float * const tau, const int &N_tar, const int &stdim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const bool &symmetric, const bool &compact, int past_idx, int future_idx)
{
	#if DEBUG
	assert (c != NULL);
	assert (!c->isNull());
	assert (c->getDim() == 4);
	assert (c->w() != NULL);
	assert (c->x() != NULL);
	assert (c->y() != NULL);
	assert (c->z() != NULL);
	assert (tau != NULL);
	assert (N_tar > 0);
	assert (stdim == 4);
	assert (manifold == DUST);
	assert (a > 0.0);
	assert (zeta < HALF_PI);
	assert (r_max > 0.0);
	assert (alpha > 0.0);
	assert (!compact);
	assert (past_idx >= 0 && past_idx < N_tar);
	assert (future_idx >= 0 && future_idx < N_tar);
	#endif

	if (future_idx < past_idx) {
		//Bitwise swap
		past_idx ^= future_idx;
		future_idx ^= past_idx;
		past_idx ^= future_idx;
	}

	IntData idata = IntData();
	idata.limit = 60;
	idata.tol = 1.0e-5;

	double x1 = pow(1.5 * tau[past_idx], 1.0 / 3.0);
	double x2 = pow(1.5 * tau[future_idx], 1.0 / 3.0);
	//printf("x1: %.8f\tx2: %.8f\n", x1, x2);
	double omega12;
	double lambda;

	bool timelike = nodesAreRelated(c, N_tar, stdim, manifold, a, zeta, zeta1, r_max, alpha, symmetric, compact, past_idx, future_idx, &omega12);
	//printf("dt: %.8f\tdx: %.8f\n", c->w(future_idx) - c->w(past_idx), omega12);
	omega12 *= alpha / a;

	//Bisection Method
	double res = 1.0, tol = 1.0E-5;
	double lower, upper;
	int iter = 0, max_iter = 10000;

	if (!timelike) {
		lower = -1.0 / (x2 * x2 * x2 * x2);
		upper = 0.0;
		//printf("Spacelike.\n");
	} else {
		lower = 0.0;
		upper = 1000.0;
		//printf("Timelike.\n");
	}

	//Use upper or lower branch of the solution
	bool upper_branch = false;

	//For use with 2F1
	HyperType ht;
	double f;
	double f_err = 1.0E-6;
	int f_nt;

	double z = POW2(POW2(x1 / x2, EXACT), EXACT);
	ht = getHyperType(z);
	f_nt = getNumTerms(ht.w, f_err);
	
	//Turning point for \tilde{omega12}
	double branch_cutoff;
	switch (ht.type) {
	case 0:
		// 0 <= z <= 0.5
		//printf("Case 0.\n");
		_2F1(0.25, 0.5, 1.25, ht.w, &f, &f_err, &f_nt, false);
		branch_cutoff = POW2(GAMMA(0.25, STL), EXACT) * x2/ SQRT(8.0 * M_PI, STL) - 2.0 * x1 * f;
		break;
	case 1:
		// 0.5 < z <= 1
		//printf("Case 1.\n");
		_2F1(1.0, 0.75, 1.5, ht.w, &f, &f_err, &f_nt, false);
		branch_cutoff = x1 * SQRT(ht.w, STL) * f;
		break;
	default:
		//This should never be reached
		//printf("Default.\n");
		return NAN;
	}
	//printf("Branch cutoff: %.8f\n", branch_cutoff);
	assert (branch_cutoff == branch_cutoff);

	if (!!branch_cutoff && fabs(omega12 - branch_cutoff) / branch_cutoff < 1.0e-3)
		omega12 = branch_cutoff;
	if (!timelike && omega12 > branch_cutoff)
		upper_branch = true;

	double x0, mx;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	while (fabs(res) > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		if (upper_branch) {
			mx = geodesicMaxX(x0);
			idata.lower = x1;
			idata.upper = mx;
			res = integrate1D(&dustLookupKernel, (void*)&x0, &idata, QAGS);
			//assert (res == res);
			if (res != res) {
				gsl_integration_workspace_free(idata.workspace);
				return INF;
			}
			idata.lower = x2;
			res += integrate1D(&dustLookupKernel, (void*)&x0, &idata, QAGS);
			//Problem recorded here
			//assert (res == res);
			if (res != res) {
				//printf("x1: %f\tx2: %f\tomega: %f\tcutoff: %f\n", x1, x2, omega12, branch_cutoff);
				//printf("lower: %f\tupper: %f\n", x2, mx);
				//printChk();
				gsl_integration_workspace_free(idata.workspace);
				return INF;
			}
			res *= 2.0;
			res -= omega12;
			if (res < 0.0)
				lower = x0;
			else
				upper = x0;
		} else {
			idata.lower = x1;
			idata.upper = x2;
			res = 2.0 * integrate1D(&dustLookupKernel, (void*)&x0, &idata, QAGS);
			//assert (res == res);
			if (res != res) {
				gsl_integration_workspace_free(idata.workspace);
				return INF;
			}
			res -= omega12;
			if (res > 0.0)
				lower = x0;
			else
				upper = x0;
		}
		iter++;
	}
	lambda = x0;
	//printf("x1: %f\tx2: %f\tlam: %f\tdx: %f\n", x1, x2, lambda, omega12);
	//printf("tau1: %f\ttau2: %f\n", tau[past_idx], tau[future_idx]);

	double distance;
	if (upper_branch) {
		idata.lower = tau[past_idx];
		idata.upper = geodesicMaxTau(manifold, lambda);
		distance = integrate1D(&dustDistKernel, (void*)&lambda, &idata, QAGS);
		//assert (distance == distance);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}

		idata.lower = tau[future_idx];
		distance += integrate1D(&dustDistKernel, (void*)&lambda, &idata, QAGS);
		//assert (distance == distance);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	} else {
		idata.lower = tau[past_idx];
		idata.upper = tau[future_idx];
		distance = integrate1D(&dustDistKernel, (void*)&lambda, &idata, QAGS);
		//assert (distance == distance);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	}

	gsl_integration_workspace_free(idata.workspace);
	//printf("distance: %f\n\n", distance);

	return distance;
}

//Returns the geodesic distance for two points on a K = 1 de Sitter manifold
inline double distanceDeSitterSph(Coordinates *c, const float * const tau, const int &N_tar, const int &stdim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const bool &symmetric, const bool &compact, int past_idx, int future_idx)
{
	#if DEBUG
	assert (c != NULL);
	assert (!c->isNull());
	assert (c->getDim() == 4);
	assert (c->w() != NULL);
	assert (c->x() != NULL);
	assert (c->y() != NULL);
	assert (c->z() != NULL);
	assert (tau != NULL);
	assert (N_tar > 0);
	assert (stdim == 4);
	assert (manifold == DE_SITTER);
	assert (a > 0.0);
	assert (zeta > 0.0);
	assert (zeta < HALF_PI);
	assert (compact);
	assert (past_idx >= 0 && past_idx < N_tar);
	assert (future_idx >= 0 && future_idx < N_tar);
	#endif

	if (future_idx < past_idx) {
		//Bitwise swap
		past_idx ^= future_idx;
		future_idx ^= past_idx;
		past_idx ^= future_idx;
	}

	IntData idata = IntData();
	idata.limit = 60;
	idata.tol = 1.0e-5;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);

	double omega12, lambda;
	bool timelike = nodesAreRelated(c, N_tar, stdim, manifold, a, zeta, zeta1, r_max, alpha, symmetric, compact, past_idx, future_idx, &omega12);
	//printf("\ndt: %f\tdx: %f\n", c->w(future_idx) - c->w(past_idx), omega12);

	//Bisection Method
	double res = 1.0, tol = 1.0e-5;
	double lower, upper;
	int iter = 0, max_iter = 10000;

	if (!timelike) {
		lower = -1.0 / (cosh(tau[future_idx]) * cosh(tau[future_idx]));
		upper = 0.0;
		//printf("Spacelike.\n");
	} else {
		lower = 0.0;
		upper = 1000.0;
		//printf("Timelike.\n");
	}

	//Check if distance is infinite
	lambda = 0.0;
	idata.lower = tau[past_idx];
	idata.upper = 200;
	double inf_dist = integrate1D(&deSitterLookupKernel, (void*)&lambda, &idata, QAGS);
	idata.lower = tau[future_idx];
	inf_dist += integrate1D(&deSitterLookupKernel, (void*)&lambda, &idata, QAGS);
	if (omega12 > inf_dist) {
		//printf("Distance: Inf\n");
		return INF;
	}
	//printf("inf_dist: %f\n", inf_dist);

	//Use upper or lower branch of the solution
	double ml = -1.0 / (cosh(tau[future_idx]) * cosh(tau[future_idx]));
	bool upper_branch = false;

	idata.lower = tau[past_idx];
	idata.upper = tau[future_idx];
	double branch_cutoff = integrate1D(&deSitterLookupKernel, (void*)&ml, &idata, QAGS);
	//printf("Branch cutoff: %f\n", branch_cutoff);
	if (!timelike && omega12 > branch_cutoff) 
		upper_branch = true;

	double x0, mt;
	while (upper - lower > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		if (!timelike && upper_branch) {
			mt = acosh(1.0 / sqrt(-1.0 * x0));
			//printf("[0] lambda: %f\tt1: %f\tt2: %f\ttm: %f\n", x0, tau[past_idx], tau[future_idx], mt);
			//res = 2.0 * deSitterLookupExact(mt, x0);
			idata.lower = tau[past_idx];
			idata.upper = mt;
			res = integrate1D(&deSitterLookupKernel, (void*)&x0, &idata, QAGS);
			assert (res == res);
			//res -= deSitterLookupExact(tau[past_idx], x0);
			idata.lower = tau[future_idx];
			res += integrate1D(&deSitterLookupKernel, (void*)&x0, &idata, QAGS);
			assert (res == res);
			//res -= deSitterLookupExact(tau[future_idx], x0);
			//assert (res == res);
			//printf("\tres = %f\n", res);
			res -= omega12;
			if (res < 0.0)
				lower = x0;
			else
				upper = x0;
		} else {
			//printf("[1] lambda: %f\tt1: %f\tt2: %f\n", x0, tau[past_idx], tau[future_idx]);
			//res = deSitterLookupExact(tau[future_idx], x0);
			idata.lower = tau[past_idx];
			idata.upper = tau[future_idx];
			res = integrate1D(&deSitterLookupKernel, (void*)&x0, &idata, QAGS);
			assert (res == res);
			//res -= deSitterLookupExact(tau[past_idx], x0);
			//assert (res == res);
			//printf("\tres = %f\n", res);
			res -= omega12;
			if (res > 0.0)
				lower = x0;
			else
				upper = x0;
		}
		iter++;
	}
	lambda = x0;

	//printf("Lambda: %f\n", lambda);
	//fflush(stdout);

	double distance;
	if (!timelike && upper_branch) {
		idata.lower = tau[past_idx];
		idata.upper = acosh(1.0 / sqrt(-1.0 * x0));
		distance = integrate1D(&deSitterDistKernel, (void*)&lambda, &idata, QAGS);

		idata.lower = tau[future_idx];
		distance += integrate1D(&deSitterDistKernel, (void*)&lambda, &idata, QAGS);
	} else {
		idata.lower = tau[past_idx];
		idata.upper = tau[future_idx];
		distance = integrate1D(&deSitterDistKernel, (void*)&lambda, &idata, QAGS);
	}

	//printf("Distance: %f\n", distance);
	//fflush(stdout);

	gsl_integration_workspace_free(idata.workspace);

	return distance;
}

//Returns the geodesic distance for two points on a K = 0 de Sitter manifold
inline double distanceDeSitterFlat(Coordinates *c, const float * const tau, const int &N_tar, const int &stdim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const bool &symmetric, const bool &compact, int past_idx, int future_idx)
{
	#if DEBUG
	assert (c != NULL);
	assert (!c->isNull());
	assert (c->getDim() == 4);
	assert (c->w() != NULL);
	assert (c->x() != NULL);
	assert (c->y() != NULL);
	assert (c->z() != NULL);
	assert (tau != NULL);
	assert (N_tar > 0);
	assert (stdim == 4);
	assert (manifold == DE_SITTER);
	assert (a > 0.0);
	assert (zeta > HALF_PI);
	assert (zeta1 > HALF_PI);
	assert (zeta > zeta1);
	assert (!compact);
	assert (past_idx >= 0 && past_idx < N_tar);
	assert (future_idx >= 0 && future_idx < N_tar);
	#endif

	if (future_idx < past_idx) {
		//Bitwise swap
		past_idx ^= future_idx;
		future_idx ^= past_idx;
		past_idx ^= future_idx;
	}

	double omega12, lambda;
	double x1 = tauToEtaFlat(tau[past_idx]);
	double x2 = tauToEtaFlat(tau[future_idx]);
	bool timelike = nodesAreRelated(c, N_tar, stdim, manifold, a, zeta, zeta1, r_max, alpha, symmetric, compact, past_idx, future_idx, &omega12);

	//Bisection Method
	double res = 1.0, tol = 1.0e-5;
	double lower, upper;
	int iter = 0, max_iter = 10000;

	if (!timelike) {
		lower = -1.0 * x2 * x2;
		upper = 0.0;
	} else {
		lower = 0.0;
		upper = 1000.0;
	}

	//Check if distance is infinite
	double inf_dist = fabs(x1) + fabs(x2);
	if (omega12 > inf_dist)
		return INF;

	//Use upper or lower branch of the solution
	bool upper_branch = false;
	double branch_cutoff = sqrt(x1 * x1 - x2 * x2);
	if (!timelike && omega12 > branch_cutoff)
		upper_branch = true;

	double x0 = 0.0;
	while (upper - lower > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		if (!timelike && upper_branch) {
			res = sqrt(x1 * x1 + x0) + sqrt(x2 * x2 + x0);
			res -= omega12;
			if (res < 0.0)
				lower = x0;
			else
				upper = x0;
		} else {
			res = sqrt(x1 * x1 + x0) - sqrt(x2 * x2 + x0);
			res -= omega12;
			if (res > 0.0)
				lower = x0;
			else
				upper = x0;
		}
		iter++;
	}
	lambda = x0;

	double distance;
	if (timelike)
		distance = asinh(sqrt(lambda) / x1) - asinh(sqrt(lambda) / x2);
	else {
		if (upper_branch)
			distance = asin(sqrt(-lambda) / x1) + asin(sqrt(-lambda) / x2) + M_PI;
		else
			distance = asin(sqrt(-lambda) / x1) - asin(sqrt(-lambda) / x2);
	}

	return distance;
}

//NOTE: This function is no longer supported or maintained
//Returns the exact distance between two nodes in 4D
//Uses a pre-generated lookup table
//O(xxx) Efficiency (revise this)
inline double distance_v1(const double * const table, const float4 &node_a, const float tau_a, const float4 &node_b, const float tau_b, const int &stdim, const Manifold &manifold, const double &a, const double &alpha, const long &size, const bool &compact)
{
	#if DEBUG
	assert (manifold == DE_SITTER || manifold == FLRW);
	if (manifold == FLRW) {
		assert (table != NULL);
		assert (alpha > 0.0);
		assert (size > 0);
		assert (!compact);
	}
	assert (stdim == 4);
	assert (a > 0.0);
	#endif

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

//Returns the embedded distance between two nodes in a 5D embedding
//O(xxx) Efficiency (revise this)
inline double distanceEmb(const float4 &node_a, const float &tau_a, const float4 &node_b, const float &tau_b, const int &stdim, const Manifold &manifold, const double &a, const double &alpha, const bool &compact)
{
	#if DEBUG
	assert (stdim == 4);
	assert (manifold == DE_SITTER || manifold == FLRW);
	//assert (compact);
	#endif

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
	} else if (manifold == DE_SITTER && compact) {
		z0_a = SINH(tau_a, APPROX ? FAST : STL);
		z0_b = SINH(tau_b, APPROX ? FAST : STL);

		z1_a = COSH(tau_a, APPROX ? FAST : STL);
		z1_b = COSH(tau_b, APPROX ? FAST : STL);
	}

	if (manifold == DE_SITTER && !compact) {
		inner_product_ab = POW2(node_a.w, EXACT) + POW2(node_b.w, EXACT);
		#if DIST_V2
		inner_product_ab -= flatProduct_v2(node_a, node_b);
		#else
		inner_product_ab -= flatProduct_v1(node_a, node_b);
		#endif
		inner_product_ab /= 2.0 * node_a.w * node_b.w;
	} else {
		#if DIST_V2
		inner_product_ab = z1_a * z1_b * sphProduct_v2(node_a, node_b) - z0_a * z0_b;
		#else
		inner_product_ab = z1_a * z1_b * sphProduct_v1(node_a, node_b) - z0_a * z0_b;
		#endif
	}

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
inline double distanceH(const float2 &hc_a, const float2 &hc_b, const int &stdim, const Manifold &manifold, const double &zeta)
{
	#if DEBUG
	assert (stdim == 2);
	assert (manifold == HYPERBOLIC);
	assert (zeta != 0.0);
	#endif

	if (hc_a.x == hc_b.x && hc_a.y == hc_b.y)
		return 0.0f;

	double dtheta = M_PI - ABS(M_PI - ABS(hc_a.y - hc_b.y, STL), STL);
	double distance = ACOSH(COSH(zeta * hc_a.x, APPROX ? FAST : STL) * COSH(zeta * hc_b.x, APPROX ? FAST : STL) - SINH(zeta * hc_a.x, APPROX ? FAST : STL) * SINH(zeta * hc_b.x, APPROX ? FAST : STL) * COS(dtheta, APPROX ? FAST : STL), APPROX ? INTEGRATION : STL, VERY_HIGH_PRECISION) / zeta;

	return distance;
}

#endif
