#ifndef COORDINATES_H_
#define COORDINATES_H_

#include "Operations.h"

/////////////////////////////
//(C) Will Cunningham 2015 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

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
	return 0.0;
}

//Returns a zenith angle
//distributed in the range [0, pi)
inline float get_zenith_angle(UGenerator &rng)
{
	return acosf(1.0 - 2.0 * rng());
}

//Returns an azimuthal uniformly
//distributed in the range [0, 2pi)
inline float get_azimuthal_angle(UGenerator &rng)
{
	return TWO_PI * rng();
}

//Returns a radius distributed in r^(d-2)
inline float get_radius(UGenerator &rng, const float &r_max, const int d)
{
	return r_max * pow(rng(), 1.0 / (d - 1));
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
inline float4 get_sph_d3(PGenerator &rng)
{
	float4 f;
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
inline float4 get_sph_d4(PGenerator &rng)
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
inline float4 get_flat_d3(UGenerator &urng, PGenerator &prng, const float &r_max)
{
	float r = get_radius(urng, r_max, 4);
	float4 f = get_sph_d3(prng);
	f.x *= r;
	f.y *= r;
	f.z *= r;
	return f;
}

//////////////////////////
// 2-D De Sitter Slab   //
// Spherical Foliation  //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for eta
inline float get_2d_asym_sph_deSitter_slab_eta(UGenerator &rng, const double &zeta)
{
	return static_cast<float>(atan(rng() / tan(zeta)));
}

//Returns a value for theta
#define get_2d_asym_sph_deSitter_slab_theta \
	get_uniform_azimuthal_angle

//Returns embedded spatial coordinates
#define get_2d_asym_sph_deSitter_slab_emb \
	get_azimuthal_emb

/////////////////////////
// 2-D De Sitter Slab  //
// Spherical Foliation //
// Symmetric About Eta //
/////////////////////////

//Returns a value for eta
inline float get_2d_sym_sph_deSitter_slab_eta(UGenerator &rng, const double &zeta)
{
	int flip = rng() < 0.5 ? 1 : -1;
	return flip * get_2d_asym_sph_deSitter_slab_eta(rng, zeta);
}

//Returns a value for theta
#define get_2d_sym_sph_deSitter_slab_theta \
	get_azimuthal_angle

//Returns embedded spatial coordinates
#define get_2d_sym_sph_deSitter_slab_emb \
	get_sph_d2

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
inline float get_2d_asym_sph_deSitter_diamond_theta(UGenerator mrng, const float &eta)
{
	return 0.0f;
}

//Returns embedded spatial coordinates
inline float2 get_2d_asym_sph_deSitter_diamond_emb(UGenerator &rng, const float &eta)
{
	float theta = get_2d_asym_sph_deSitter_diamond_theta(rng, eta);
	return make_float2(cosf(theta), sinf(theta));
}

//////////////////////////
// 4-D De Sitter Slab   //
// Spherical Foliation  //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for eta
inline float get_4d_asym_sph_deSitter_slab_eta(UGenerator &rng)
{
	return 0.0f;
}

//Returns a value for theta1
#define get_4d_asym_sph_deSitter_slab_theta1 \
	get_hyperzenith_angle

//Returns a value for theta2
#define get_4d_asym_sph_deSitter_slab_theta2 \
	get_zenith_angle

//Returns a value for theta3
#define get_4d_asym_sph_deSitter_slab_theta3 \
	get_azimuthal_angle

//Returns the embedded spatial coordinates
#define get_4d_asym_sph_deSitter_slab_emb \
	get_sph_d4

/////////////////////////
// 4-D De Sitter Slab  //
// Spherical Foliation //
// Symmetric About Eta //
/////////////////////////

//Returns a value for eta
inline float get_4d_sym_sph_deSitter_slab_eta(UGenerator &rng)
{
	int flip = rng() < 0.5 ? 1 : -1;
	return flip * get_4d_asym_sph_deSitter_slab_eta(rng);
}

//Returns a value for theta1
#define get_4d_sym_sph_deSitter_slab_theta1 \
	get_hyperzenith_angle

//Returns a value for theta2
#define get_4d_sym_sph_deSitter_slab_theta2 \
	get_zenith_angle

//Returns a value for theta3
#define get_4d_sym_sph_deSitter_slab_theta3 \
	get_azimuthal_angle

//Returns the embedded spatial coordinates
#define get_4d_sym_sph_deSitter_slab_emb \
	get_sph_d4

///////////////////////////
// 4-D De Sitter Diamond //
// Spherical Foliation   //
// Asymmetric About Eta  //
///////////////////////////

//Returns a value for eta
inline float get_4d_asym_sph_deSitter_diamond_eta(UGenerator &rng)
{
	return 0.0;
}

//Returns a value for theta1 given eta
inline float get_4d_asym_sph_deSitter_diamond_theta1(UGenerator &rng)
{
	return 0.0;
}

//Returns a value for theta2
#define get_4d_asym_sph_deSitter_diamond_theta2 \
	get_zenith_angle

//Returns a value for theta3
#define get_4d_asym_sph_deSitter_diamond_theta3 \
	get_azimuthal_angle

//Returns the embedded spatial coordinates
inline float4 get_4d_asym_sph_deSitter_diamond_emb(UGenerator &urng, PGenerator &prng)
{
	float theta1 = get_4d_asym_sph_deSitter_diamond_theta1(urng);
	float4 f = get_sph_d3(prng);
	f.w = cosf(theta1);
	f.x *= sinf(theta1);
	f.y *= sinf(theta1);
	f.z *= sinf(theta1);
	return f;
}

//////////////////////////
// 4-D De Sitter Slab   //
// Flat Foliation       //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for eta
inline float get_4d_asym_flat_deSitter_slab_eta(UGenerator &rng, const double &eta_min, const double &eta_max)
{
	return static_cast<float>(eta_min * pow(1.0 - rng() * (1.0 - POW3(eta_min / eta_max, EXACT)), -1.0 / 3.0));
}

//Returns a value for radius
#define get_4d_asym_flat_deSitter_slab_radius(rng, r_max) \
	get_radius(rng, r_max, 4)

//Returns a value for theta2
#define get_4d_asym_flat_deSitter_slab_theta2 \
	get_zenith_angle

//Returns a value for theta3
#define get_4d_asym_flat_deSitter_slab_theta3 \
	get_azimuthal_angle

//Returns the Cartesian coordinates
#define get_4d_asym_flat_deSitter_slab_cartesian \
	get_flat_d3

///////////////////////////
// 4-D De Sitter Diamond //
// Flat Foliation        //
// Asymmetric About Eta  //
///////////////////////////

//Returns a value for eta
inline float get_4d_asym_flat_deSitter_diamond_eta(UGenerator &rng)
{
	return 0.0f;
}

//Returns a value for radius given eta
inline float get_4d_asym_flat_deSitter_diamond_radius(UGenerator &rng)
{
	return 0.0f;
}

//Returns a value for theta2
#define get_4d_asym_flat_deSitter_diamond_theta2 \
	get_zenith_angle

//Returns a value for theta3
#define get_4d_asym_flat_deSitter_diamond_theta3 \
	get_azimuthal_angle

//Returns the Cartesian coordinates
inline float4 get_4d_asym_flat_deSitter_diamond_cartesian(UGenerator &urng, PGenerator &prng)
{
	float r = get_4d_asym_flat_deSitter_diamond_radius(urng);
	float4 f = get_sph_d3(prng);
	f.x *= r;
	f.y *= r;
	f.z *= r;
	return f;
}

//////////////////////////
// 4-D Dust Slab        //
// Flat Curvature       //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for eta
inline float get_4d_asym_flat_dust_slab_eta(UGenerator &rng)
{
	return 0.0f;
}

//Returns a value for radius
#define get_4d_asym_flat_dust_slab_radius(rng, r_max) \
	get_radius(rng, r_max, 4)

//Returns a value for theta2
#define get_4d_asym_flat_dust_slab_theta2 \
	get_zenith_angle

//Returns a value for theta3
#define get_4d_asym_flat_dust_slab_theta3 \
	get_azimuthal_angle

//Returns the Cartesian coordinates
#define get_4d_asym_flat_dust_slab_cartesian \
	get_flat_d3

//////////////////////////
// 4-D FLRW Slab        //
// Positive Curvature   //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for tau
inline float get_4d_asym_sph_flrw_slab_tau(UGenerator &rng)
{
	return 0.0f;
}

//Returns a value for theta1
#define get_4d_asym_sph_flrw_slab_theta1 \
	get_hyperzenith_angle

//Returns a value for theta2
#define get_4d_asym_sph_flrw_slab_theta2 \
	get_zenith_angle

//Returns a value for theta3
#define get_4d_asym_sph_flrw_slab_theta3 \
	get_azimuthal_angle

//Returns the Cartesian coordinates
#define get_4d_asym_sph_flrw_slab_cartesian \
	get_sph_d4

//////////////////////////
// 4-D FLRW Slab        //
// Flat Curvature       //
// Asymmetric About Eta //
//////////////////////////

//Returns a value for tau
#define get_4d_asym_flat_flrw_slab_tau \
	get_4d_asym_sph_flrw_slab_tau

//Returns a value for radius
#define get_4d_asym_flat_flrw_slab_radius(rng, r_max) \
	get_radius(rng, r_max, 4)

//Returns a value for theta2
#define get_4d_asym_flat_flrw_slab_theta2 \
	get_zenith_angle

//Returns a value for theta3
#define get_4d_asym_flat_flrw_slab_theta3 \
	get_azimuthal_angle

//Returns the Cartesian coordinates
#define get_4d_asym_flat_flrw_slab_cartesian \
	get_flat_d3

#endif
