/////////////////////////////
//(C) Will Cunningham 2015 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef COORDINATES_H_
#define COORDINATES_H_

#include "Operations.h"

// The purpose of this file is to easily allow the addition of new
// manifolds, regions, dimensions, and foliations.  There are many redundant
// definitions simply to improve readability and avoid gratuitous
// if/else blocks which are difficult to extend.

///////////////////////////////
// General Purpose Functions //
///////////////////////////////

//Returns the first polar angle
//on a hypersphere in the range [0, pi)
inline float get_hyperzenith_angle(UGenerator &rng)
{
	double theta;
	double x = HALF_PI;
	double r = rng();
	if (!newton(&solve_hyperzenith, &x, 1000, TOL, &r, NULL, NULL))
		theta = NAN;
	else
		theta = x;

	#if DEBUG
	assert (theta == theta);
	#endif

	return theta;
}

//Returns a zenith angle
//distributed in the range [0, pi)
inline float get_zenith_angle(UGenerator &rng)
{
	return acos(1.0 - 2.0 * rng());
}

//Returns an azimuthal uniformly
//distributed in the range [0, 2pi)
inline float get_azimuthal_angle(UGenerator &rng)
{
	return TWO_PI * rng();
}

//Returns a radius distributed in r^(d-2)
inline float get_radius(UGenerator &rng, const float r_max, const float d)
{
	return r_max * pow(rng(), 1.0 / (d - 1.0));
}

//Returns (x,y) uniformly
//distributed on a unit circle
inline float2 get_sph_d2(UGenerator &rng)
{
	float theta = get_azimuthal_angle(rng);
	return make_float2(cosf(theta), sinf(theta));
}

//Returns (x,y,z) uniformly distributed
//on the surface of a sphere
inline float3 get_sph_d3(NGenerator &rng)
{
	float3 f;
	f.x = rng();
	f.y = rng();
	f.z = rng();

	float d = 1.0f / sqrtf(f.x * f.x + f.y * f.y + f.z * f.z);
	f.x *= d;
	f.y *= d;
	f.z *= d;

	return f;
}

//Returns (w,x,y,z) uniformly distributed
//on the surface of a hypersphere
inline float4 get_sph_d4(NGenerator &rng)
{
	float4 f;
	f.w = rng();
	f.x = rng();
	f.y = rng();
	f.z = rng();

	float d = 1.0f / sqrtf(f.w * f.w + f.x * f.x + f.y * f.y + f.z * f.z);
	f.w *= d;
	f.x *= d;
	f.y *= d;
	f.z *= d;

	return f;
}

//Returns (x,y,z) for a point uniformly
//distributed inside a sphere of radius r_max
inline float3 get_flat_d3(UGenerator &urng, NGenerator &nrng, const float r_max)
{
	float r = get_radius(urng, r_max, 4.0);
	float3 f = get_sph_d3(nrng);
	f.x *= r;
	f.y *= r;
	f.z *= r;
	return f;
}

/////////////////////////
// 2-D Minkowski Slab  //
// Flat Curvature      //
// Symmetric About Eta //
/////////////////////////

inline float get_2d_sym_flat_minkowski_slab_eta(UGenerator &rng, const double eta0)
{
	return (2.0 * eta0 * rng()) - eta0;
}

inline float get_2d_sym_flat_minkowski_slab_radius(UGenerator &rng, const double r_max)
{
	return (2.0 * r_max * rng()) - r_max;
}

/////////////////////////
// 2-D Minkowski Slab  //
// Positive Curvature  //
// Symmetric About Eta //
/////////////////////////

#define get_2d_sym_sph_minkowski_slab_eta(rng, eta0) \
	get_2d_sym_flat_minkowski_slab_eta(rng, eta0)

#define get_2d_sym_sph_minkowski_slab_theta(rng) \
	get_azimuthal_angle(rng)

///////////////////////////
// 2-D Minkowski Diamond //
// Flat Curvature        //
// Asymmetric About Eta  //
///////////////////////////

inline float get_2d_asym_flat_minkowski_diamond_u(UGenerator &rng, const double xi)
{
	return rng() * xi;
}

#define get_2d_asym_flat_minkowski_diamond_v(rng, xi) \
	get_2d_asym_flat_minkowski_diamond_u(rng, xi)

//////////////////////////
// 2-D De Sitter Slab   //
// Spherical Foliation  //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for eta
inline float get_2d_asym_sph_deSitter_slab_eta(UGenerator &rng, const double eta0)
{
	double eta = atan(rng() * tan(eta0));
	if ((float)eta >= eta0)
		eta = eta0 - 1.0e-6;
	return eta;
}

//Returns a value for theta
#define get_2d_asym_sph_deSitter_slab_theta(rng) \
	get_azimuthal_angle(rng)

//Returns embedded spatial coordinates
#define get_2d_asym_sph_deSitter_slab_emb(rng) \
	get_sph_d2(rng)

/////////////////////////
// 2-D De Sitter Slab  //
// Spherical Foliation //
// Symmetric About Eta //
/////////////////////////

//Returns a value for eta
inline float get_2d_sym_sph_deSitter_slab_eta(UGenerator &rng, const double eta0)
{
	int flip = rng() < 0.5 ? 1 : -1;
	return flip * get_2d_asym_sph_deSitter_slab_eta(rng, eta0);
}

//Returns a value for theta
#define get_2d_sym_sph_deSitter_slab_theta(rng) \
	get_azimuthal_angle(rng)

//Returns embedded spatial coordinates
#define get_2d_sym_sph_deSitter_slab_emb(rng) \
	get_sph_d2(rng)

//////////////////////////
// 2-D De Sitter Slab   //
// Hyperbolic Foliation //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for tau
inline float get_2d_asym_hyp_deSitter_slab_tau(UGenerator &rng, const double tau0)
{
	return acosh(rng() * (cosh(tau0) - 1.0) + 1.0);
}

//Returns a value for theta
#define get_2d_asym_hyp_deSitter_slab_theta(rng) \
	get_azimuthal_angle(rng)

///////////////////////////
// 2-D De Sitter Diamond //
// Spherical Foliation   //
// Asymmetric About Eta  //
///////////////////////////

//Returns a value for eta
inline float get_2d_asym_sph_deSitter_diamond_eta(UGenerator &rng)
{
	return 0.0f;
}

//Returns a value for theta given eta
inline float get_2d_asym_sph_deSitter_diamond_theta(UGenerator mrng, const float eta)
{
	return 0.0f;
}

//Returns embedded spatial coordinates
inline float2 get_2d_asym_sph_deSitter_diamond_emb(UGenerator &rng, const float eta)
{
	float theta = get_2d_asym_sph_deSitter_diamond_theta(rng, eta);
	return make_float2(cosf(theta), sinf(theta));
}

/////////////////////////
// 2-D Hyperbolic Slab //
// Spherical Foliation //
// Asymmetric About R  //
/////////////////////////

//Returns a value for radius
inline float get_2d_asym_sph_hyperbolic_slab_radius(UGenerator &rng, const double r_max, const double zeta)
{
	return 2.0 * zeta * acosh(rng() * (cosh(r_max / (2.0 * zeta)) - 1.0f) + 1.0f);
}

//Non-Uniform Radial Distribution
//Use this for a Growing H2 Model
inline float get_2d_asym_sph_hyperbolic_slab_nonuniform_radius(UGenerator &rng, const double r_max, const double zeta)
{
	return 4.0 * zeta * asinh(sqrt(rng() * POW2(sinh(r_max / (4.0 * zeta)))));
}

//Returns a value for theta
#define get_2d_asym_sph_hyperbolic_slab_theta(rng) \
	get_azimuthal_angle(rng)

//////////////////////////
// 2-D Polycone Slab    //
// Positive Curvature   //
// Asymmetric About Tau //
//////////////////////////

//Returns a value for tau
#define get_2d_asym_sph_polycone_slab_tau(rng, tau0, mu) \
	get_radius(rng, tau0, mu + 2.0)

//Returns a value for theta
#define get_2d_asym_sph_polycone_slab_theta(rng) \
	get_azimuthal_angle(rng)

/////////////////////////
// 3-D Minkowski Slab  //
// Flat Curvature      //
// Symmetric About Eta //
/////////////////////////

//Returns a value for eta
inline float get_3d_sym_flat_minkowski_slab_eta(UGenerator &rng, const double eta0)
{
	return eta0 * (2.0 * rng() - 1.0);
}

//Returns a value for radius
#define get_3d_sym_flat_minkowski_slab_radius(rng, r_max) \
	get_radius(rng, r_max, 3.0)

//Returns a value for azimuthal angle
#define get_3d_sym_flat_minkowski_slab_theta(rng) \
	get_azimuthal_angle(rng)

//////////////////////////
// 3-D Minkowski Cube   //
// Flat Curvature       //
// Asymmetric About Eta //
//////////////////////////

//Return a value for eta
inline float get_3d_asym_flat_minkowski_cube_eta(UGenerator &rng, const double eta0)
{
	return rng() * eta0;
}

//Return a value for x
inline float get_3d_asym_flat_minkowski_cube_x(UGenerator &rng, const double r_max)
{
	return rng() * r_max;
}

#define get_3d_asym_flat_minkowski_cube_y(rng, r_max) \
	get_3d_asym_flat_minkowski_cube_x(rng, r_max)

//////////////////////////
// 4-D De Sitter Slab   //
// Spherical Foliation  //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for eta
inline float get_4d_asym_sph_deSitter_slab_eta(UGenerator &rng, const double zeta)
{
	double eta;
	double x = 0.2;
	double r = rng();
	double a = r * (2.0 + POW2(1.0 / sin(zeta))) / tan(zeta);
	if (!newton(&solve_eta_12836_0, &x, 1000, TOL, &a, NULL, NULL))
		eta = NAN;
	else
		eta = 2.0 * atan(x);

	#if DEBUG
	assert (eta == eta);
	#endif

	return eta;
}

//Returns a value for theta1
#define get_4d_asym_sph_deSitter_slab_theta1(rng) \
	get_hyperzenith_angle(rng)

//Returns a value for theta2
#define get_4d_asym_sph_deSitter_slab_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_asym_sph_deSitter_slab_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the embedded spatial coordinates
#define get_4d_asym_sph_deSitter_slab_emb(nrng) \
	get_sph_d4(nrng)

/////////////////////////
// 4-D De Sitter Slab  //
// Spherical Foliation //
// Symmetric About Eta //
/////////////////////////

//Returns a value for eta
inline float get_4d_sym_sph_deSitter_slab_eta(UGenerator &rng, const double zeta)
{
	int flip = rng() < 0.5 ? 1 : -1;
	return flip * get_4d_asym_sph_deSitter_slab_eta(rng, zeta);
}

//Returns a value for theta1
#define get_4d_sym_sph_deSitter_slab_theta1(rng) \
	get_hyperzenith_angle(rng)

//Returns a value for theta2
#define get_4d_sym_sph_deSitter_slab_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_sym_sph_deSitter_slab_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the embedded spatial coordinates
#define get_4d_sym_sph_deSitter_slab_emb(nrng) \
	get_sph_d4(nrng)

///////////////////////////
// 4-D De Sitter Diamond //
// Spherical Foliation   //
// Asymmetric About Eta  //
///////////////////////////

//Returns a value for u
inline float get_4d_asym_sph_deSitter_diamond_u(UGenerator &rng, const double xi, const double mu)
{
	double u = 0.3;
	double rmu = rng() * mu;
	if (!bisection(&solve_u_13348_0_bisec, &u, 2000, 0.0, xi, TOL, true, &rmu, NULL, NULL)) 
		u = NAN;

	#if DEBUG
	assert (u == u);
	#endif

	return u;
}

//Returns a value for v given u
//This function can by OPTIMIZED by passing constants
inline float get_4d_asym_sph_deSitter_diamond_v(UGenerator &rng, const double u)
{
	double v = 0.05;
	double p[2];
	p[0] = u;
	p[1] = rng();
	//if (!newton(&solve_v_13348_0, &v, 1000, TOL, p, NULL, NULL))
	if (!bisection(&solve_v_13348_0_bisec, &v, 2000, 0.0, u, TOL, true, p, NULL, NULL))
		v = NAN;

	#if DEBUG
	assert (v == v);
	#endif

	return v;
}

//Returns a value for theta2
#define get_4d_asym_sph_deSitter_diamond_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_asym_sph_deSitter_diamond_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the embedded spatial coordinates
inline float4 get_4d_asym_sph_deSitter_diamond_emb(UGenerator &urng, NGenerator &nrng, const double u, const double v)
{
	float theta1 = (u - v) / sqrt(2.0);
	float3 f = get_sph_d3(nrng);
	float4 g;
	g.w = cosf(theta1);
	g.x = f.x * sinf(theta1);
	g.y = f.y * sinf(theta1);
	g.z = f.z * sinf(theta1);
	return g;
}

//////////////////////////
// 4-D De Sitter Slab   //
// Flat Foliation       //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for eta
inline float get_4d_asym_flat_deSitter_slab_eta(UGenerator &rng, const double eta_min, const double eta_max)
{
	return eta_min * pow(1.0 - rng() * (1.0 - POW3(eta_min / eta_max)), -1.0 / 3.0);
}

//Returns a value for radius
#define get_4d_asym_flat_deSitter_slab_radius(rng, r_max) \
	get_radius(rng, r_max, 4.0)

//Returns a value for theta2
#define get_4d_asym_flat_deSitter_slab_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_asym_flat_deSitter_slab_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the Cartesian coordinates
#define get_4d_asym_flat_deSitter_slab_cartesian(urng, nrng, r_max) \
	get_flat_d3(urng, nrng, r_max)

///////////////////////////
// 4-D De Sitter Diamond //
// Flat Foliation        //
// Asymmetric About Eta  //
///////////////////////////

//Returns a value for u
inline float get_4d_asym_flat_deSitter_diamond_u(UGenerator &rng, const double xi, const double mu)
{
	double x = -exp(-(mu * rng() + 1.0));
	double w = gsl_sf_lambert_W0(x);

	return 2.0 * xi * (sqrt(1.0 + w) - 1.0) / w - xi;
}

//Returns a value for v given u
//This function can by OPTIMIZED by passing constants
inline float get_4d_asym_flat_deSitter_diamond_v(UGenerator &rng, const double u, const double xi)
{
	double t1 = POW3(u);
	double t2 = -3.0 * POW2(u) * xi;
	double t3 = 3.0 * u * POW2(xi);
	double t4 = -POW3(xi);

	double F = rng();
	double alpha = (t1 + t3) * (F - 2.0) + (t2 + t4) * F;
	double beta = 3.0 * u * ((t1 + t3) * F + (t2 + t4) * (F - 2.0));
	double gamma = 3.0 * u * alpha;
	double delta = POW2(beta) - POW2(gamma);
	double C = pow(POW2(beta + gamma) * (beta - gamma), 1.0 / 3.0);

	return -(beta + C + delta / C) / (3.0 * alpha);
}

//Returns a value for theta2
#define get_4d_asym_flat_deSitter_diamond_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_asym_flat_deSitter_diamond_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the Cartesian coordinates
#define get_4d_asym_flat_deSitter_diamond_cartesian(urng, nrng, u, v) \
	get_flat_d3(urng, nrng, (u - v) / sqrt(2.0))

//////////////////////////
// 4-D Dust Slab        //
// Flat Curvature       //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for eta
inline float get_4d_asym_flat_dust_slab_tau(UGenerator &rng, const double tau0)
{
	return tau0 * pow(rng(), 1.0 / 3.0); 
}

//Returns a value for radius
#define get_4d_asym_flat_dust_slab_radius(rng, r_max) \
	get_radius(rng, r_max, 4.0)

//Returns a value for theta2
#define get_4d_asym_flat_dust_slab_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_asym_flat_dust_slab_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the Cartesian coordinates
#define get_4d_asym_flat_dust_slab_cartesian(urng, nrng, r_max) \
	get_flat_d3(urng, nrng, r_max)

//////////////////////////
// 4-D Dust Diamond     //
// Flat Curvature       //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for u
inline float get_4d_asym_flat_dust_diamond_u(UGenerator &rng, const double xi)
{
	return xi * pow(rng(), 1.0 / 12.0);
}

//Returns a value for v given u
//This function can by OPTIMIZED by passing constants
inline float get_4d_asym_flat_dust_diamond_v(UGenerator &rng, const double u)
{
	double v = 0.2;
	double p[2];
	p[0] = u;
	p[1] = rng();
	if (!bisection(&solve_v_11332_0_bisec, &v, 2000, 0.0, u, TOL, true, p, NULL, NULL))
		v = NAN;

	#if DEBUG
	assert (v == v);
	#endif

	return v;
}

//Returns a value for theta2
#define get_4d_asym_flat_dust_diamond_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_asym_flat_dust_diamond_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the Cartesian coordinates
#define get_4d_asym_flat_dust_diamond_cartesian(urng, nrng, u, v) \
	get_flat_d3(urng, nrng, (u - v) / sqrt(2.0))

//////////////////////////
// 4-D FLRW Slab        //
// Positive Curvature   //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for tau
inline float get_4d_asym_sph_flrw_slab_tau(UGenerator &rng, const double tau0)
{
	double tau = 0.5;
	double p[2];
	p[0] = tau0;
	p[1] = rng();

	if (tau0 > 1.8) {	//The value 1.8 is from trial/error
		if (!bisection(&solve_tau_10884_0_bisec, &tau, 2000, 0.0, tau0, TOL, true, p, NULL, NULL))
			tau = NAN;
	} else {
		if (!newton(&solve_tau_10884_0, &tau, 1000, TOL, p, NULL, NULL))
			tau = NAN;
	}

	assert (tau == tau);

	return tau;
}

//Returns a value for theta1
#define get_4d_asym_sph_flrw_slab_theta1(rng) \
	get_hyperzenith_angle(rng)

//Returns a value for theta2
#define get_4d_asym_sph_flrw_slab_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_asym_sph_flrw_slab_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the Cartesian coordinates
#define get_4d_asym_sph_flrw_slab_cartesian(nrng) \
	get_sph_d4(nrng)

//////////////////////////
// 4-D FLRW Slab        //
// Flat Curvature       //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for tau
#define get_4d_asym_flat_flrw_slab_tau(rng, tau0) \
	get_4d_asym_sph_flrw_slab_tau(rng, tau0)

//Returns a value for radius
#define get_4d_asym_flat_flrw_slab_radius(rng, r_max) \
	get_radius(rng, r_max, 4.0)

//Returns a value for theta2
#define get_4d_asym_flat_flrw_slab_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_asym_flat_flrw_slab_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the Cartesian coordinates
#define get_4d_asym_flat_flrw_slab_cartesian(urng, nrng, r_max) \
	get_flat_d3(urng, nrng, r_max)

//////////////////////////
// 4-D FLRW Diamond     //
// Flat Curvature       //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for tau
//It's easier to re-implement the bisection method here
//  due to the structure of inc/Operations.h,
//  and because the kernel here is an integral rather
//  than a closed-form expression
inline float get_4d_asym_flat_flrw_diamond_tau(UGenerator &rng, IntData * const idata, double params[], const double tau0, const double tau_half, const double lower_prob, const double mu, const double mu1)
{
	double p = rng();
	double res = 1.0, tol = 1.0e-8, x0 = 0.0;
	double lower, upper;
	double lhs = p * mu;
	int iter = 0, max_iter = 10000;
	bool use_lower = (p <= lower_prob);

	if (use_lower) {
		lower = 0.0;
		upper = tau_half;
		idata->lower = 0.0;
	} else {
		lower = tau_half;
		upper = tau0;
		idata->lower = tau_half;
		lhs -= mu1;
	}

	while (upper - lower > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		if (use_lower) {
			idata->upper = x0;
			res = integrate1D(&volume_11396_0_lower, params, idata, QAG);
		} else {
			idata->upper = x0;
			res = integrate1D(&volume_11396_0_upper, params, idata, QAG);
		}
		#if DEBUG
		assert (res == res);
		#endif
		res -= lhs;
		if (res < 0.0)
			lower = x0;
		else
			upper = x0;
		iter++;
	}

	return x0;
}

//Returns a value for radius given tau
inline float get_4d_asym_flat_flrw_diamond_radius(UGenerator &rng, const double eta, const double zeta)
{
	double r, r_max;
	if (eta < (HALF_PI - zeta) / 2.0)
		r_max = eta;
	else
		r_max = (HALF_PI - zeta) - eta;
	r = get_radius(rng, r_max, 4.0);
	return r;
}

//Returns a value for theta2
#define get_4d_asym_flat_flrw_diamond_theta2(rng) \
	get_zenith_angle(rng)

//Returns a value for theta3
#define get_4d_asym_flat_flrw_diamond_theta3(rng) \
	get_azimuthal_angle(rng)

//Returns the Cartesian coordinates
#define get_4d_asym_flat_flrw_diamond_cartesian(urng, nrng, r) \
	get_flat_d3(urng, nrng, r)

#endif
