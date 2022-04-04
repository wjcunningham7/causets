/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include "Causet.h"
#include "Subroutines.h"

//======================//
// Root-Finding Kernels //
//======================//

inline double eta_12836_0(const double &x, const double &a)
{
	return -a + x * (6.0 + x * (3.0 * a + x * (-4.0 + x * (-3.0 * a + x * (6.0 + a * x)))));
}

inline double eta_prime_12836_0(const double &x, const double &a)
{
	return 6.0 * (1.0 + x * (a + x * (-2.0 + x * (-2.0 * a + x * (5.0 + a * x)))));
}

inline double tau_10884_0(const double &x, const double &tau0, const double &rval)
{
	return (sinh(3.0 * x) - 3.0 * x) / (sinh(3.0 * tau0) - 3.0 * tau0) - rval;
}

inline double tau_prime_10884_0(const double &x, const double &tau0)
{
	return 6.0 * POW2(sinh(1.5 * x)) / (sinh(3.0 * tau0) - 3.0 * tau0);
}

inline double hyperzenith(const double &x, const double &rval)
{
	return (2.0 * x - sin(2.0 * x)) / TWO_PI - rval;
}

inline double hyperzenith_prime(const double &x)
{
	return POW2(sin(x)) / HALF_PI;
}

inline double u_13348_0(const double &x, const double &rmu)
{
	return (log(0.5 * (1.0 / cos(sqrt(2.0) * x) + 1.0)) - 1.0 / POW2(cos(x / sqrt(2.0))) + 1.0) - rmu;
}

inline double u_prime_13348_0(const double &x)
{
	return sqrt(2.0) * POW3(tan(x / sqrt(2.0))) / cos(sqrt(2.0) * x);
}

inline double v_13348_0(const double &x, const double &u, const double &rval)
{
	return (0.25 * cos(sqrt(2.0) * u) / sin(u / sqrt(2.0))) * (4.0 * (1.0 + 3.0 * cos(sqrt(2.0) * u) + cos(2.0 * sqrt(2.0) * u)) / sin(u / sqrt(2.0)) + (sin((2.0 * u - 3.0 * x) / sqrt(2.0)) + 3.0 * sin(x / sqrt(2.0)) + 3.0 * sin((x - 2.0 * u) / sqrt(2.0)) - sin(3.0 * (2.0 * u + x) / sqrt(2.0)) - 3.0 * sin((4.0 * u + x) / sqrt(2.0)) + sin((2.0 * u + 3.0 * x) / sqrt(2.0))) / (POW2(tan(u / sqrt(2.0))) * POW3(cos((u + x) / sqrt(2.0))))) - rval;
}

inline double v_prime_13348_0(const double &x, const double &u)
{
	return (-3.0 / (4.0 * sqrt(2.0))) * cos(sqrt(2.0) * u) * (-2.0 * cos(u / sqrt(2.0)) + cos((u - 2.0 * x) / sqrt(2.0)) + cos((3.0 * u - 2.0 * x) / sqrt(2.0))) / (POW2(tan(u / sqrt(2.0))) * sin(u / sqrt(2.0)) * POW2(POW2(cos((u + x) / sqrt(2.0)))));
}

inline double v_11332_0(const double &x, const double &u, const double &rval)
{
	double u2 = u * u;
	double u3 = u2 * u;
	double u4 = u3 * u;
	double u5 = u4 * u;
	double u6 = u5 * u;
	double u7 = u6 * u;
	double u8 = u7 * u;
	double u9 = u8 * u;
	double u10 = u9 * u;
	double u11 = u10 * u;

	return x * (u10 + x * (3.0 * u9 + x * (13.0 * u8 / 3.0 + x * (2.0 * u7 + x * (-14.0 * u6 / 5.0 + x * (-14.0 * u5 / 3.0 + x * (-2.0 * u4 + x * (u3 + x * (13.0 * u2 / 9.0 + x * (0.6 * u + x / 11.0)))))))))) - 1981.0 * u11 * rval / 495.0;
}

inline double v_prime_11332_0(const double &x, const double &u)
{
	double u2 = u * u;
	double u3 = u2 * u;
	double u4 = u3 * u;
	double u5 = u4 * u;
	double u6 = u5 * u;
	double u7 = u6 * u;
	double u8 = u7 * u;
	double u9 = u8 * u;
	double u10 = u9 * u;

	return u10 + x * (6.0 * u9 + x * (13.0 * u8 + x * (8.0 * u7 + x * (-14.0 * u6 + x * (-28.0 * u5 + x * (-14.0 * u4 + x * (8.0 * u3 + x * (13.0 * u2 + x * (6.0 * u + x)))))))));
}

//This function gives the indefinite integral form of the volume
//Use volume = f(1.5) - f(-1.5) to calculate the volume
inline double volume_77834_1(double x)
{
	#if DEBUG
	assert (fabs(x) <= 1.5); 
	#endif

	double t1 = 4.0 * x;
	double t2 = x * sqrt(1.0 + 4.0 * POW2(x) / 3.0);
	double t3 = 0.5 * sqrt(3.0) * asinh(2.0 * x / sqrt(3.0));

	return t1 - t2 - t3;
}

//This function gives the integral of the volume from -r_max to -r_max
inline double volume_75499530_2(const double eta0, const double r_max)
{
	#if DEBUG
	assert (eta0 <= r_max);
	#endif

	double beta = eta0 + r_max;
	double t1 = sqrt(4.0 * POW2(r_max) - POW2(beta));
	double t2 = log((2.0 * r_max + t1) / beta);

	return r_max * POW2(beta) * t2 / t1;
}

inline double v_A04000C000_4(const double x, const double u, const double rval)
{
	return -rval * POW3(u) + 3.0 * POW2(u) * x - 3.0 * u * POW2(x) + POW3(x);
}

inline double mmdim(const double dim, const double rval)
{
	#if DEBUG
	assert (dim > 0.0);
	assert (rval > 0.0);
	#endif

	if (dim < TOL) return dim;
	return (GAMMA(dim + 1.0) * GAMMA(0.5 * dim) / (4.0 * GAMMA(1.5 * dim))) - rval;
}

//Returns eta Residual
//Used in 3+1 K = 1 Asymmetric de Sitter Slab
inline double solve_eta_12836_0(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);	//"a" (not same as pseudoradius)
	#endif

	return -1.0 * eta_12836_0(x, p1[0]) / eta_prime_12836_0(x, p1[0]);
}

//Returns tau Residual
//Used in 3+1 Flat Asymmetric FLRW Slab
inline double solve_tau_10884_0(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);			//tau0
	assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	#endif

	return -1.0 * tau_10884_0(x, p1[0], p1[1]) / tau_prime_10884_0(x, p1[0]);
}

//Returns tau Residual in Bisection Algorithm
//Used in 3+1 Flat Asymmetric FLRW Slab
inline double solve_tau_10884_0_bisec(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);			//tau0
	assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	#endif

	return tau_10884_0(x, p1[0], p1[1]);
}

//Returns theta1 Residual
//Used in causets with a "hyperzenith" angle
inline double solve_hyperzenith(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0 && p1[0] < 1.0);	//rval
	#endif

	return -1.0 * hyperzenith(x, p1[0]) / hyperzenith_prime(x);
}

//Returns u Residual
//Used in 3+1 K = 1 de Sitter Diamond (Asymmetric)
inline double solve_u_13348_0(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);	//mu * rval
	#endif

	return -1.0 * u_13348_0(x, p1[0]) / u_prime_13348_0(x);
}

//Returns u Residual in Bisection Algorithm
//Used in 3+1 K = 1 de Sitter Diamond (Asymmetric)
inline double solve_u_13348_0_bisec(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);	//mu * rval
	#endif

	return u_13348_0(x, p1[0]);
}

//Returns v Residual
//Used in 3+1 K = 1 de Sitter Diamond (Asymmetric)
inline double solve_v_13348_0(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);			//u
	assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	#endif

	return -1.0 * v_13348_0(x, p1[0], p1[1]) / v_prime_13348_0(x, p1[0]);
}

//Returns v Residual in Bisection Algorithm
//Used in 3+1 K = 1 de Sitter Diamond (Asymmetric)
inline double solve_v_13348_0_bisec(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);			//u
	assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	#endif

	return v_13348_0(x, p1[0], p1[1]);
}

//Returns v Residual
//Used in 3+1 Flat Dust Diamond (Asymmetric)
inline double solve_v_11332_0(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);			//u
	assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	#endif

	return -1.0 * v_11332_0(x, p1[0], p1[1]) / v_prime_11332_0(x, p1[0]);
}

//Returns v Residual in Bisection Algorithm
//Used in 3+1 Flat Dust Diamond (Asymmetric)
inline double solve_v_11332_0_bisec(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);			//u
	assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	#endif

	return v_11332_0(x, p1[0], p1[1]);
}

//Returns v Residual in Bisection Algorithm
//Used in Flat 3+1 Minkowski Diamond
inline double solve_v_A04000C000_4_bisec(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);			//u
	assert (p1[1] > 0.0 && p1[1] < 1.0);	//rval
	#endif

	return v_A04000C000_4(x, p1[0], p1[1]);
}

//Returns dimension Residual in Bisection Algorithm
//Used for Myrheim Meyer Dimension
inline double solve_mmdim_bisec(const double x, const double * const p1, const float * const p2, const int * const p3)
{
	#if DEBUG
	assert (p1 != NULL);
	assert (p1[0] > 0.0);	//rval
	#endif

	return mmdim(x, p1[0]);
}

//=========================//
// Spatial Length Formulae //
//=========================//

//X1 Coordinate of de Sitter 3-Metric
inline float X1_SPH(const float &theta1)
{
	return cosf(theta1);
}

//X2 Coordinate of Spherical 3-Metric
inline float X2_SPH(const float &theta1, const float &theta2)
{
	return sinf(theta1) * cosf(theta2);
}

//X3 Coordinate of Spherical 3-Metric
inline float X3_SPH(const float &theta1, const float &theta2, const float &theta3)
{
	return sinf(theta1) * sinf(theta2) * cosf(theta3);
}

//X4 Coordinate of Spherical 3-Metric
inline float X4_SPH(const float &theta1, const float &theta2, const float &theta3)
{
	return sinf(theta1) * sinf(theta2) * sinf(theta3);
}

//X Coordinate from Flat 3-Metric
inline float X_FLAT(const float &theta1, const float &theta2, const float &theta3)
{
	return theta1 * sinf(theta2) * cosf(theta3);
}

//Y Coordinate from Flat 3-Metric
inline float Y_FLAT(const float &theta1, const float &theta2, const float &theta3)
{
	return theta1 * sinf(theta2) * sinf(theta3);
}

//Z Coordinate from Flat 3-Metric
inline float Z_FLAT(const float &theta1, const float &theta2)
{
	return theta1 * cosf(theta2);
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
inline float flatProduct_v2(const float5 &sc0, const float5 &sc1)
{
	return POW2(sc0.w) + POW2(sc1.w) -
	       2.0f * sc0.w * sc1.w * (cosf(sc0.x) * cosf(sc1.x) +
	       sinf(sc0.x) * sinf(sc1.x) * (cosf(sc0.y) * cosf(sc1.y) +
	       sinf(sc0.y) * sinf(sc1.y) * cosf(sc0.z - sc1.z)));
}

inline float flatProduct_v2(const float4 &sc0, const float4 &sc1)
{
	return POW2(sc0.x) + POW2(sc1.x) -
	       2.0f * sc0.x * sc1.x * (cosf(sc0.y) * cosf(sc1.y) +
	       sinf(sc0.y) * sinf(sc1.y) * cosf(sc0.z - sc1.z));
}

inline float flatProduct_v2(const float3 &sc0, const float3 &sc1)
{
	return POW2(sc0.y) + POW2(sc1.y) - 2.0f * sc0.y * sc1.y * cosf(sc0.z - sc1.z);
}

//Embedded form
inline float flatEmbProduct(const float5 &sc0, const float5 &sc1)
{
	return POW2(sc0.x - sc1.x) + POW2(sc0.y - sc1.y) + POW2(sc0.z - sc1.z);
}

//==========================//
// Node Relation Algorithms //
//==========================//

//Assumes coordinates have been temporally ordered
//Used for relations in Lorentzian spaces
inline bool nodesAreRelated(Coordinates *c, const Spacetime spacetime, const int N, const double a, const double zeta, const double zeta1, const double r_max, const double alpha, int past_idx, int future_idx, double *omega12)
{
	#if DEBUG
	assert (!c->isNull());
	assert (spacetime.stdimIs("2") || spacetime.stdimIs("3") || spacetime.stdimIs("4") || spacetime.stdimIs("5"));
	assert (spacetime.manifoldIs("Minkowski") || spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW") || spacetime.manifoldIs("Polycone"));

	if (spacetime.stdimIs("2"))
		assert (c->getDim() == 2);
	else if (spacetime.stdimIs("4")) {
		assert (c->getDim() == 4);
		assert (c->w() != NULL);
		assert (c->z() != NULL);
	}

	assert (c->x() != NULL);
	assert (c->y() != NULL);

	assert (N > 0);
	assert (a > 0.0);
	if (spacetime.manifoldIs("De_Sitter")) {
		if (spacetime.curvatureIs("Positive")) {
			assert (zeta > 0.0);
			assert (zeta < HALF_PI);
		} else if (spacetime.curvatureIs("Flat")) {
			assert (zeta > HALF_PI);
			assert (zeta1 > HALF_PI);
			assert (zeta > zeta1);
		}
	} else if (spacetime.manifoldIs("Dust") || spacetime.manifoldIs("FLRW")) {
		assert (zeta < HALF_PI);
		assert (alpha > 0.0);
	}
	if (spacetime.curvatureIs("Flat"))
		assert (r_max > 0.0);
		
	assert (past_idx >= 0 && past_idx < N);
	assert (future_idx >= 0 && future_idx < N);
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
	if (spacetime.stdimIs("2") || spacetime.stdimIs("3"))
		dt = fabs(c->x(future_idx) - c->x(past_idx));
	else if (spacetime.stdimIs("4"))
		#if EMBED_NODES
		dt = fabs(c->v(future_idx) - c->v(past_idx));
		#else
		dt = fabs(c->w(future_idx) - c->w(past_idx));
		#endif
	else if (spacetime.stdimIs("5"))
		dt = fabs(c->v(future_idx) - c->v(past_idx));

	#if DEBUG
	assert (dt >= 0.0f);
	/*if (spacetime.curvatureIs("Flat") && spacetime.manifoldIs("De_Sitter"))
		assert (dt <= zeta - zeta1);
	else {
		if (spacetime.symmetryIs("Temporal"))
			assert (dt <= 2.0f * static_cast<float>(HALF_PI - zeta));
		else
			assert (dt <= static_cast<float>(HALF_PI - zeta));
	}*/
	#endif

	//Spatial Interval
	if (spacetime.stdimIs("2")) {
		if (spacetime.curvatureIs("Positive") || spacetime.curvatureIs("Negative")) {
			#if EMBED_NODES
			float phi1 = atan2(c->z(future_idx), c->y(future_idx));
			float phi2 = atan2(c->z(past_idx), c->y(past_idx));
			dx = M_PI - fabs(M_PI - fabs(phi2 - phi1));
			#else
			dx = M_PI - fabs(M_PI - fabs(c->y(future_idx) - c->y(past_idx)));
			#endif
		} else
			dx = fabs(c->y(future_idx) - c->y(past_idx));
	} else if (spacetime.stdimIs("3")) {
		if (spacetime.curvatureIs("Flat"))
			dx = sqrtf(flatProduct_v2(c->getFloat3(past_idx), c->getFloat3(future_idx)));
		else if (spacetime.curvatureIs("Positive")) {
			float dX = M_PI - fabs(M_PI - fabs(c->y(future_idx) - c->y(past_idx)));
			float dY = M_PI - fabs(M_PI - fabs(c->z(future_idx) - c->z(past_idx)));
			dx = sqrtf(POW2(dX) + POW2(dY));
		}
	} else if (spacetime.stdimIs("4")) {
		if (spacetime.curvatureIs("Positive")) {
			//Spherical Law of Cosines
			#if EMBED_NODES
			dx = acosf(sphEmbProduct(c->getFloat5(past_idx), c->getFloat5(future_idx)));
			#else
			#if DIST_V2
			dx = acosf(sphProduct_v2(c->getFloat4(past_idx), c->getFloat4(future_idx)));
			#else
			dx = acosf(sphProduct_v1(c->getFloat4(past_idx), c->getFloat4(future_idx)));
			#endif
			#endif
		} else if (spacetime.curvatureIs("Flat")) {
			//Euclidean Law of Cosines
			#if EMBED_NODES
			dx = sqrtf(flatEmbProduct(c->getFloat5(past_idx), c->getFloat5(future_idx)));
			#else
			#if DIST_V2
			dx = sqrtf(flatProduct_v2(c->getFloat4(past_idx), c->getFloat4(future_idx)));
			#else
			dx = sqrtf(flatProduct_v1(c->getFloat4(past_idx), c->getFloat4(future_idx)));
			#endif
			#endif
		}
	} else if (spacetime.stdimIs("5"))
		dx = sqrtf(flatProduct_v2(c->getFloat5(past_idx), c->getFloat5(future_idx)));

	if (omega12 != NULL)
		*omega12 = dx;

	if (dx < dt)
		return true;
	else
		return false;
}

//Assumes coordinates have been temporally ordered
//Used for relations in Hyperbolic spaces
inline bool nodesAreRelatedHyperbolic(const Node &nodes, const Spacetime &spacetime, const int N, const double zeta, const double r_max, const bool link_epso, int past_idx, int future_idx, double *product)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (spacetime.stdimIs("2"));
	assert (spacetime.manifoldIs("Hyperbolic"));
	assert (spacetime.curvatureIs("Positive"));

	assert (nodes.crd->getDim() == 2);
	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.id.tau != NULL);

	assert (N > 0);
	assert (zeta > 0.0);
	assert (r_max > 0.0);
	assert (past_idx >= 0 && past_idx < N);
	assert (future_idx >= 0 && future_idx < N);
	assert (past_idx != future_idx);
	#endif

	float inner_product = 0.0f, max_r = 0.0f;

	if (future_idx < past_idx) {
		//Bitwise swap
		past_idx ^= future_idx;
		future_idx ^= past_idx;
		past_idx ^= future_idx;
	}

	//Normalized Inner Product
	inner_product = cosh(nodes.id.tau[past_idx]) * cosh(nodes.id.tau[future_idx]) - sinh(nodes.id.tau[past_idx]) * sinh(nodes.id.tau[future_idx]) * cos(nodes.crd->y(past_idx) - nodes.crd->y(future_idx));

	if (product != NULL)
		*product = inner_product;
	
	if (link_epso) {
		double m = 5.0;
		double R_i = nodes.id.tau[future_idx] + 2.0 * log(m * M_PI / nodes.id.tau[future_idx]);
		double R_j = nodes.id.tau[past_idx] + 2.0 * log(m * M_PI / nodes.id.tau[past_idx]);
		max_r = std::max(R_i, R_j);
	} else
		max_r = r_max;

	if (acosh(inner_product) < max_r)
		return true;
	else
		return false;
}

inline void deSitterInnerProduct(const Node &nodes, const Spacetime &spacetime, const int N, int past_idx, int future_idx, double *product)
{
	#if DEBUG
	assert (!nodes.crd->isNull());
	assert (spacetime.stdimIs("2"));
	assert (spacetime.manifoldIs("De_Sitter"));
	assert (spacetime.curvatureIs("Positive"));

	assert (nodes.crd->getDim() == 2);
	assert (nodes.crd->x() != NULL);
	assert (nodes.crd->y() != NULL);
	assert (nodes.id.tau != NULL);
	assert (product != NULL);

	assert (N > 0);
	assert (past_idx >= 0 && past_idx < N);
	assert (future_idx >= 0 && future_idx < N);
	assert (past_idx != future_idx);
	#endif

	*product = sinh(nodes.id.tau[past_idx]) * sinh(nodes.id.tau[future_idx]) - cosh(nodes.id.tau[past_idx]) * cosh(nodes.id.tau[future_idx]) * cos(nodes.crd->y(past_idx) - nodes.crd->y(future_idx));
}

//Check if point is inside asymmetric diamond
inline bool iad(const float eta, const float x, const double eta_min, const double eta_max)
{
	return eta > eta_min && ((eta < (eta_max + eta_min) / 2.0 && fabs(x) < eta - eta_min) || (eta > (eta_max + eta_min) / 2.0 && fabs(x) < eta_max - eta));
}

//Check if point is inside symmetric diamond
inline bool isd(const float eta, const float x, const float eta0)
{
	//return eta > -eta0 && ((eta < 0.0 && fabs(x) < -eta0 - eta) || (eta > 0.0 && fabs(x) < eta0 - eta));
	return fabs(eta) < eta0 && fabs(x) < eta0 - fabs(eta);
}

//=================================//
// Conformal/Cosmic Time Relations //
//=================================//

//-----------------------//
// Formulae in de Sitter //
//-----------------------//

//Conformal to Rescaled Time (de Sitter, spherical foliation)
inline double etaToTauSph(const double eta)
{
	#if DEBUG
	assert (fabs(eta) < HALF_PI);
	#endif

	return SGN(eta) * acosh(1.0 / cos(eta));
}

//Conformal to Rescaled Time (de Sitter flat foliation)
inline double etaToTauFlat(const double eta)
{
	#if DEBUG
	assert (eta < 0.0);
	#endif

	return -log(-eta);
}

//Rescaled to Conformal Time (de Sitter spherical foliation)
inline double tauToEtaSph(const double tau)
{
	return SGN(tau) * acos(1.0 / cosh(tau));
}

//Rescaled to Conformal Time (de Sitter flat foliation)
inline double tauToEtaFlat(const double tau)
{
	#if DEBUG
	assert (tau > 0.0);
	#endif

	return -exp(-tau);
}

//Rescaled to Conformal time (de Sitter hyperbolic foliation)
inline double tauToEtaHyp(const double tau)
{
	#if DEBUG
	assert (tau > 0.0);
	#endif

	return log(tanh(tau / 2.0));
}

//-------------------//
// Formulae for Dust //
//-------------------//

//Conformal to Rescaled Time (Dust)
inline double etaToTauDust(const double eta, const double a, const double alpha)
{
	#if DEBUG
	assert (eta > 0.0);
	assert (a > 0.0);
	assert (alpha > 0.0);
	#endif

	return POW3(eta * alpha / a) / 12.0;
}

//Rescaled to Conformal Time (Dust)
inline double tauToEtaDust(const double tau, const double a, const double alpha)
{
	#if DEBUG
	assert (tau > 0.0);
	assert (a > 0.0);
	assert (alpha > 0.0);
	#endif

	return pow(12.0 * tau, 1.0 / 3.0) * a / alpha;
}

//------------------//
// Formulae in FLRW //
//------------------//

//For use with GNU Scientific Library
inline double tauToEtaFLRW(double tau, void *params)
{
	#if DEBUG
	assert (tau > 0.0);
	#endif

	return pow(sinh(1.5 * tau), (-2.0 / 3.0));
}

//'Exact' Solution (Hypergeometric Series)
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
	double z = 1.0 / POW2(cosh(1.5 * tau));
	double w;
	if (z >= 0.0 && z <= 0.5) {
		w = z;
		_2F1(1.0 / 3.0, 5.0 / 6.0, 4.0 / 3.0, w, &f, &err, &nterms, false);
		eta = sqrt(3.0 * POW3(M_PI)) / (GAMMA(5.0 / 6.0) * GAMMA(-4.0 / 3.0)) - pow(w, 1.0 / 3.0) * f;
	} else if (z > 0.5 && z <= 1.0) {
		w = 1 - z;
		_2F1(0.5, 1.0, 7.0 / 6.0, w, &f, &err, &nterms, false);
		eta = 2.0 * pow(z * sqrt(w), 1.0 / 3.0) * f;
	} else
		//This should never be reached
		return NAN;

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

	double g = 9.0 * GAMMA(2.0 / 3.0) * alpha * eta / a;
	g -= 4.0 * sqrt(3.0) * pow(M_PI, 1.5) / GAMMA(5.0 / 6.0);
	g /= 3.0 * GAMMA(-1.0 / 3.0);

	#if DEBUG
	assert (g > 0.0);
	#endif

	return g;
}

//-----------------------//
// Formulae for Polycone //
//-----------------------//

inline double etaToTauPolycone(const double eta, const double a, const double gamma)
{
	#if DEBUG
	assert (eta < 0.0);
	assert (a > 0.0);
	assert (gamma > 2.0);
	#endif

	double g = gamma / (gamma - 2.0);
	return pow((1 - g) * eta / pow(a, -g), 1.0 / (1.0 - g));
}

inline double tauToEtaPolycone(const double tau, const double a, const double gamma)
{
	#if DEBUG
	assert (tau > 0.0);
	assert (a > 0.0);
	assert (gamma > 2.0);
	#endif

	double g = gamma / (gamma - 2.0);
	return pow(a, -g) * pow(tau, 1.0 - g) / (1.0 - g);
}

//=====================================//
// Integration Supplementary Functions //
//=====================================//

//Approximates (108) in [2]
inline double isf_01(double &r)
{
	double zeta = 0.0;
	double err = 0.0;
	double f;
	int nterms = 10;

	if (fabs(r - 1.0) < 0.05)
		nterms = 20;

	if (r < 1.0) {
		double z = -1.0 * POW3(r);
		_2F1(1.0 / 6.0, 0.5, 7.0 / 6.0, z, &f, &err, &nterms, false);
		zeta = 2.0 * sqrt(r) * f;
	} else {
		double z = -1.0 / POW3(r);
		_2F1(1.0 / 3.0, 0.5, 4.0 / 3.0, z, &f, &err, &nterms, false);
		zeta = sqrt(4.0 / M_PI) * GAMMA(7.0 / 6.0) * GAMMA(1.0 / 3.0) - f / r;
	}

	return zeta;
}

//Gives rescaled FLRW scale factor as a function of eta
//Uses 'ctuc' lookup table
inline double isf_02(double *table, double size, double eta, double a, double alpha)
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
		
	return pow(sinh(1.5 * tau), 2.0 / 3.0);
}

//=================//
// Volume Formulae //
//=================//

//--------------------------------//
// Volume in 1+1 Minkowski Saucer //
//--------------------------------//

//This function gives the boundary value eta(x)
inline double eta_77834_1(double x, double eta0)
{
	#if DEBUG
	#if SPECIAL_SAUCER
	assert (fabs(x) <= 1.5);
	#else
	assert (fabs(x) <= sqrt(eta0 * (2.0 - eta0)));
	#endif
	#endif

	#if SPECIAL_SAUCER
	return 2.0 - sqrt(4.0 * POW2(x) / 3.0 + 1.0);
	#else
	return 1.0 - sqrt(POW2(x) + POW2(1.0 - eta0));
	#endif
}

inline double r_240800011_6(double t, double K)
{
	#if DEBUG
	assert (fabs(t) <= 0.5);
	assert (K >= 0.0);
	#endif

	return sqrt(0.25 + pow(K, -2.0)) - sqrt(POW2(t) + pow(K, -2.0));
}

//-----------------------------------//
// Volume in 1+1 Minkowski Slab (S1) //
//-----------------------------------//

//This function gives the boundary value eta(x)
inline double eta_75499530_2(double x, double eta0, double r_max)
{
	#if DEBUG
	assert (fabs(x) <= r_max);
	#endif

	double beta = r_max + eta0;
	return sqrt(-(POW2(beta / r_max) - 4.0) * POW2(x) + POW2(beta)) - r_max;
}

//---------------------------------------//
// Volume in 1+1 Minkowski Triangle (T1) //
//---------------------------------------//

//This function gives the boundary value eta(x)
inline double eta_76546058_2(double x, double eta0, double r_max)
{
	#if DEBUG
	assert (r_max <= eta0);
	assert (x >= 0.0 && x <= r_max);
	#endif

	return eta0 * (1.0 - x / r_max);
}

//---------------------------------//
// Volume in 3+1 Flat FLRW Diamond //
//---------------------------------//

//This kernel is used in numerical integration
//This gives the volume from tau=0 to tau=tau0/2 in a diamond
//This result should be multiplied by (4pi/3)a^4
inline double volume_11396_0_lower(double tau, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (((double*)params)[0] > 0.0);	//tau0
	assert (((double*)params)[1] > 0.0);	//eta0
	assert (((double*)params)[2] > 0.0 && ((double*)params)[2] < ((double*)params)[0]);	//tau1/2
	assert (tau > 0.0 && tau < ((double*)params)[2]);
	#endif

	double t = sinh(1.5 * tau);
	double eta = tauToEtaFLRWExact(tau, 1.0, 1.0);

	return POW2(t) * POW3(eta);
}

//This kernel is used in numerical integration
//This gives the volume from tau=tau0/2 to tau=tau0 in a diamond
//This result should be multiplied by (4pi/3)a^4
inline double volume_11396_0_upper(double tau, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (((double*)params)[0] > 0.0);	//tau0
	assert (((double*)params)[1] > 0.0);	//eta0
	assert (((double*)params)[2] > 0.0 && ((double*)params)[2] < ((double*)params)[0]);	//tau1/2
	assert (tau > ((double*)params)[2] && tau < ((double*)params)[0]);
	#endif

	double t = sinh(1.5 * tau);
	double eta = tauToEtaFLRWExact(tau, 1.0, 1.0);

	return POW2(t) * POW3(((double*)params)[1] - eta);
}

//=========================//
// Average Degree Formulae //
//=========================//

// Note functions are written as
// > averageDegree_X_Y
// where X is the spacetime ID
// for version Y of the program

//--------------------------------------------------------//
// Rescaled Average Degree in Flat (K = 0) de Sitter Slab //
//--------------------------------------------------------//

//This kernel is used in numerical integration
//This result should be multipled by 4pi*(eta0*eta1)^3/(eta1^3 - eta0^3)
inline double averageDegree_10788_0(int dim, double x[], double *params)
{
	#if DEBUG
	assert (dim > 0);
	assert (x[0] < 0.0 && x[0] > -1.0 * HALF_PI);
	assert (x[1] < 0.0 && x[1] > -1.0 * HALF_PI);
	#endif

	//Identify x[0] with eta' coordinate
	//Identify x[1] with eta'' coordinate
	
	double t = POW2(POW2(x[0] * x[1]));
	return fabs(POW3(x[0] - x[1])) / t;
}

//---------------------------------------//
// Rescaled Average Degree in Dusty Slab //
//---------------------------------------//

//This kernel is used in numerical integration
//This result should be multiplied by (108pi / tau0^3)
inline double averageDegree_10820_0(int dim, double x[], double *params)
{
	#if DEBUG
	assert (dim > 0);
	assert (x[0] > 0.0);
	assert (x[1] > 0.0);
	#endif

	//Identify x[0] with the tau' coordinate
	//Identify x[1] with the tau'' coordinate
	
	double h1 = x[0] * x[1];
	double h2 = fabs(pow(x[0], 1.0 / 3.0) - pow(x[1], 1.0 / 3.0));

	return POW2(h1) * POW3(h2);
}

//--------------------------------------------------//
// Rescaled Average Degree in Non-Compact FLRW Slab //
//--------------------------------------------------//

//This is a kernel used in numerical integration
//Note to get the (non-compact) rescaled average degree this result must still be
//multiplied by 8pi / (sinh(3tau0)-3tau0)
inline double averageDegree_10884_0(int dim, double x[], double *params)
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

	double s1 = POW2(sinh(1.5 * x[0]));
	double s2 = POW2(sinh(1.5 * x[1]));

	return s1 * s2 * fabs(POW3(h2 - h1));
}

//----------------------------------------------//
// Rescaled Average Degree in Compact FLRW Slab //
//----------------------------------------------//

//This is a kernel used in numerical integration
//Note to get the (compact) rescaled averge degree this result must still be
//multiplied by 8pi/(sinh(3tau0)-3tau0)
//This kernel is an ALTERNATE (PRIMARY) method to averageDegreeFLRW
inline double averageDegree_12932_0(int dim, double x[], double *params)
{
	#if DEBUG
	assert (dim > 0);
	assert (x[0] > 0.0);
	assert (x[1] > 0.0);
	#endif

	//Identify x[0] with x coordinate
	//Identify x[1] with r coordinate

	double z;

	z = POW3(fabs(isf_01(x[0]) - isf_01(x[1]))) * POW2(x[0]) * POW3(x[1]) * sqrt(x[1]);
	z /= (sqrt(1.0 + 1.0 / POW3(x[0])) * sqrt(1.0 + POW3(x[1])));

	#if DEBUG
	assert (z > 0.0);
	#endif

	return z;
}

//This is a kernel used in numerical integration
//Note to get the average degree this result must still be
//multipled by (4pi/3)*delta*alpha^4/averageDegreeNorm
//This kernel is an ALTERNATE (SECONDARY) method to rescaledDegreeFLRW
inline double averageDegree_12932_0_a(int dim, double x[], double *params)
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

	z = POW3(fabs(x[0] - x[1]));
	z *= POW2(POW2(isf_02(&params[3], params[2], x[0], params[0], params[1])));
	z *= POW2(POW2(isf_02(&params[3], params[2], x[1], params[0], params[1])));

	#if DEBUG
	assert (z > 0.0);
	#endif

	return z;
}

//For use with GNU Scientific Library
inline double averageDegree_12932_0_b(double eta, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (eta > 0.0);
	#endif

	//Identify params[0] with a
	//Identify params[1] with alpha
	//Identify params[2] with size
	//Identify params[3] with table
	
	return POW2(POW2(isf_02(&((double*)params)[3], ((double*)params)[2], eta, ((double*)params)[0], ((double*)params)[1])));
}

//================//
// Action Formula //
//================//

//Calculate the action from the abundancy intervals
//The parameter 'lk' is taken to be expressed in units of the graph discreteness length 'l'
inline double calcAction(const uint64_t * const cardinalities, const int stdim, const double &epsilon)
{
	#if DEBUG
	assert (cardinalities != NULL);
	assert (stdim >= 2 && stdim <= 5);
	assert (epsilon > 0.0 && epsilon <= 1.0);
	#endif

	long double action = 0.0;

	if (epsilon < 1.0) {
		long double eps1 = epsilon / (1.0 - epsilon);
		long double eps2 = eps1 * eps1;
		long double eps3 = eps2 * eps1;
		long double epsi = 1.0;
		long double c3_1 = 27.0 / 8.0;
		long double c3_2 = 9.0 / 8.0;
		long double c4_1 = 4.0 / 3.0;
		long double ni;
		uint64_t i;

		for (i = 0; i < cardinalities[0] - 3; i++) {
			ni = static_cast<long double>(cardinalities[i+1]);
			if (stdim == 2)
				action += ni * epsi * (1.0 - 2.0 * eps1 * i + 0.5 * eps2 * i * (i - 1.0));
			else if (stdim == 3)
				action += ni * epsi * (1.0 - c3_1 * i * eps1 + c3_2 * i * (i - 1.0) * eps2);
			else if (stdim == 4)
				action += ni * epsi * (1.0 - 9.0 * eps1 * i + 8.0 * eps2 * i * (i - 1.0) - c4_1 * eps3 * i * (i - 1.0) * (i - 2.0));
			else if (stdim == 5)
				action = 0.0;
			else
				action = NAN;
			epsi *= (1.0 - epsilon);
		}

		if (stdim == 2)
			action = 2.0 * epsilon * (cardinalities[0] - 2.0 * epsilon * action);
		else if (stdim == 3)
			action = pow(M_PI / (3.0 * sqrt(2.0)), 2.0 / 3.0) / GAMMA(5.0 / 3.0) * (pow(epsilon, 2.0 / 3.0) * cardinalities[0] - pow(epsilon, 5.0 / 3.0) * action);
		else if (stdim == 4)
			action = (4.0 / sqrt(6.0)) * (sqrt(epsilon) * cardinalities[0] - pow(epsilon, 1.5) * action);
		else if (stdim == 5)
			action = 0.0;
		else
			action = NAN;
	} else {
		if (stdim == 2)
			action = 2.0 * (cardinalities[0] - 2.0 * (cardinalities[1] - 2.0 * cardinalities[2] + cardinalities[3]));
		else if (stdim == 3)
			action = pow(M_PI / (3.0 * sqrt(2.0)), 2.0 / 3.0) / GAMMA(5.0 / 3.0) * (cardinalities[0] - cardinalities[1] + (27.0 / 8.0) * cardinalities[2] - (9.0 / 4.0) * cardinalities[3]);
		else if (stdim == 4)
			action = (4.0 / sqrt(6.0)) * (cardinalities[0] - cardinalities[1] + 9.0 * cardinalities[2] - 16.0 * cardinalities[3] + 8.0 * cardinalities[4]);
		else if (stdim == 5)
			action = (pow(POW2(M_PI) / (20.0 * sqrt(2.0)), 0.4) / GAMMA(1.4)) * (cardinalities[0] - (3.0 / 8.0) * cardinalities[1] + (215.0 / 16.0) * cardinalities[2] - (225.0 / 8.0) * cardinalities[3] + (125.0 / 8.0) * cardinalities[4]);
		else
			action = NAN;
	}

	return static_cast<double>(action);
}

//////////////////////////
// Monte Carlo Analysis //
//////////////////////////

inline double specific_heat(double *data, size_t stride, size_t nvals, double beta)
{
	double variance = gsl_stats_variance(data, stride, nvals);
	return beta * beta * variance;
}

#endif
