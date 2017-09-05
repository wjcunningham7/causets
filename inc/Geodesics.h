/////////////////////////////
//(C) Will Cunningham 2015 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef GEODESICS_H_
#define GEODESICS_H_

#include "Causet.h"
#include "CuResources.h"
#include "Operations.h"

//Maximum Time in Geodesic (non-embedded)
//Returns tau_max=f(lambda) with lambda < 0
inline double geodesicMaxTau(const int manifold, const double &lambda)
{
	#if DEBUG
	assert (!strcmp(Spacetime::manifolds[manifold], "De_Sitter") || !strcmp(Spacetime::manifolds[manifold], "Dust") || !strcmp(Spacetime::manifolds[manifold], "FLRW"));
	assert (lambda < 0.0);
	#endif

	if (!strcmp(Spacetime::manifolds[manifold], "FLRW"))
		return (2.0 / 3.0) * asinh(pow(-lambda, -0.75));
	else if (!strcmp(Spacetime::manifolds[manifold], "Dust"))
		return (2.0 / 3.0) * pow(-lambda, -0.75);
	else if (!strcmp(Spacetime::manifolds[manifold], "De Sitter")) {
		double g = pow(-lambda, -0.5);
		return g >= 1.0 ? acosh(g) : 0.0;
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

	return pow(-1.0 * lambda, -0.25);
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

	return sqrt(1.0 + (x / (POW3(alpha_tilde) + POW3(x))));
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

	double lcx2 = lambda * POW2(cosh(x));
	double distance = sqrt(fabs(lcx2 / (1.0 + lcx2)));

	return distance;
}

inline double dustDistKernel(double x, void *params)
{
	#if DEBUG
	assert (params != NULL);
	assert (x >= 0.0);
	#endif

	double lambda = ((double*)params)[0];

	double g43 = lambda * pow(1.5 * x, 4.0 / 3.0);
	double distance = sqrt(fabs(g43 / (1.0 + g43)));

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

	double x4 = POW2(POW2(x));
	double omega12 = 1.0 / sqrt(1.0 + lambda * x4);

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

//=====================//
// Distance Algorithms //
//=====================//

//Returns the distance between two nodes in the non-compact K = 0 FLRW manifold
inline double distanceFLRW(Coordinates *c, const float * const tau, const Spacetime &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, int past_idx, int future_idx)
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
	assert (spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("FLRW"));
	assert (a > 0.0);
	assert (zeta < HALF_PI);
	assert (r_max > 0.0);
	assert (alpha > 0.0);
	assert (spacetime.curvatureIs("Flat"));
	assert (past_idx >= 0 && past_idx < N_tar);
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
	double omega12;
	double lambda;

	bool timelike = nodesAreRelated(c, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, past_idx, future_idx, &omega12);
	omega12 *= alpha / a;

	//Bisection Method
	double res = 1.0, tol = 1.0E-5;
	double lower, upper;
	int iter = 0, max_iter = 10000;

	if (!timelike) {
		lower = -1.0 / (x2 * x2 * x2 * x2);
		upper = 0.0;
	} else {
		lower = 0.0;
		upper = 1000.0;
	}

	//Check if distance is infinite
	double inf_dist = 4.0 * tgamma(1.0 / 3.0) * tgamma(7.0 / 6.0) / sqrt(M_PI) - tauToEtaFLRWExact(tau[past_idx], 1.0, 1.0) - tauToEtaFLRWExact(tau[future_idx], 1.0, 1.0);
	if (omega12 > inf_dist)
		return INF;

	//Use upper or lower branch of the solution
	double ml = -1.0 / (x2 * x2 * x2 * x2);
	bool upper_branch = false;

	idata.lower = x1;
	idata.upper = x2;
	idata.workspace = gsl_integration_workspace_alloc(idata.nintervals);
	double branch_cutoff;
	if (c->w(future_idx) - c->w(past_idx) > 1.0E-14) {
		branch_cutoff = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&ml, &idata, QAGS);
		if (branch_cutoff != branch_cutoff) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	} else
		branch_cutoff = 0.0;
	if (!!branch_cutoff && fabs(omega12 - branch_cutoff) / branch_cutoff < 1.0e-3)
		omega12 = branch_cutoff;
	if (!timelike && omega12 > branch_cutoff)
		upper_branch = true;

	double x0, mx;
	while (fabs(res) > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		if (!timelike && upper_branch) {
			mx = geodesicMaxX(x0);
			idata.lower = x1;
			idata.upper = mx;
			res = integrate1D(&flrwLookupKernelX, (void*)&x0, &idata, QAGS);
			if (res != res) {
				gsl_integration_workspace_free(idata.workspace);
				return INF;
			}
			idata.lower = x2;
			res += integrate1D(&flrwLookupKernelX, (void*)&x0, &idata, QAGS);
			if (res != res) {
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
			res = 2.0 * integrate1D(&flrwLookupKernelX, (void*)&x0, &idata, QAGS);
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

	double distance;
	if (!timelike && upper_branch) {
		idata.lower = tau[past_idx];
		idata.upper = geodesicMaxTau(spacetime.get_manifold(), lambda);
		distance = integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}

		idata.lower = tau[future_idx];
		distance += integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	} else {
		idata.lower = tau[past_idx];
		idata.upper = tau[future_idx];
		distance = integrate1D(&flrwDistKernel, (void*)&lambda, &idata, QAGS);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	}

	gsl_integration_workspace_free(idata.workspace);

	return distance;
}

//Returns the geodesic distance for two points on a K = 0 dust manifold
inline double distanceDust(Coordinates *c, const float * const tau, const Spacetime &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, int past_idx, int future_idx)
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
	assert (spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("Dust"));
	assert (a > 0.0);
	assert (zeta < HALF_PI);
	assert (r_max > 0.0);
	assert (alpha > 0.0);
	assert (spacetime.curvatureIs("Flat"));
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
	double omega12;
	double lambda;

	bool timelike = nodesAreRelated(c, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, past_idx, future_idx, &omega12);
	omega12 *= alpha / a;

	//Bisection Method
	double res = 1.0, tol = 1.0E-5;
	double lower, upper;
	int iter = 0, max_iter = 10000;

	if (!timelike) {
		lower = -1.0 / (x2 * x2 * x2 * x2);
		upper = 0.0;
	} else {
		lower = 0.0;
		upper = 1000.0;
	}

	//Use upper or lower branch of the solution
	bool upper_branch = false;

	//For use with 2F1
	HyperType ht;
	double f;
	double f_err = 1.0E-6;
	int f_nt;

	double z = POW2(POW2(x1 / x2));
	ht = getHyperType(z);
	f_nt = getNumTerms(ht.w, f_err);
	
	//Turning point for \tilde{omega12}
	double branch_cutoff;
	switch (ht.type) {
	case 0:
		// 0 <= z <= 0.5
		_2F1(0.25, 0.5, 1.25, ht.w, &f, &f_err, &f_nt, false);
		branch_cutoff = POW2(GAMMA(0.25)) * x2/ sqrt(8.0 * M_PI) - 2.0 * x1 * f;
		break;
	case 1:
		// 0.5 < z <= 1
		_2F1(1.0, 0.75, 1.5, ht.w, &f, &f_err, &f_nt, false);
		branch_cutoff = x1 * sqrt(ht.w) * f;
		break;
	default:
		//This should never be reached
		return NAN;
	}
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
			if (res != res) {
				gsl_integration_workspace_free(idata.workspace);
				return INF;
			}
			idata.lower = x2;
			res += integrate1D(&dustLookupKernel, (void*)&x0, &idata, QAGS);
			if (res != res) {
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

	double distance;
	if (upper_branch) {
		idata.lower = tau[past_idx];
		idata.upper = geodesicMaxTau(spacetime.get_manifold(), lambda);
		distance = integrate1D(&dustDistKernel, (void*)&lambda, &idata, QAGS);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}

		idata.lower = tau[future_idx];
		distance += integrate1D(&dustDistKernel, (void*)&lambda, &idata, QAGS);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	} else {
		idata.lower = tau[past_idx];
		idata.upper = tau[future_idx];
		distance = integrate1D(&dustDistKernel, (void*)&lambda, &idata, QAGS);
		if (distance != distance) {
			gsl_integration_workspace_free(idata.workspace);
			return INF;
		}
	}

	gsl_integration_workspace_free(idata.workspace);

	return distance;
}

//Returns the geodesic distance for two points on a K = 1 de Sitter manifold
inline double distanceDeSitterSph(Coordinates *c, const float * const tau, const Spacetime &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, int past_idx, int future_idx)
{
	#if DEBUG
	assert (!c->isNull());
	assert (c->getDim() == 4);
	assert (c->w() != NULL);
	assert (c->x() != NULL);
	assert (c->y() != NULL);
	assert (c->z() != NULL);
	assert (tau != NULL);
	assert (N_tar > 0);
	assert (spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("De_Sitter"));
	assert (a > 0.0);
	assert (zeta > 0.0);
	assert (zeta < HALF_PI);
	assert (spacetime.curvatureIs("Positive"));
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
	bool timelike = nodesAreRelated(c, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, past_idx, future_idx, &omega12);

	//Bisection Method
	double res = 1.0, tol = 1.0e-5;
	double lower, upper;
	int iter = 0, max_iter = 10000;

	if (!timelike) {
		lower = -1.0 / (cosh(tau[future_idx]) * cosh(tau[future_idx]));
		upper = 0.0;
	} else {
		lower = 0.0;
		upper = 1000.0;
	}

	//Check if distance is infinite
	lambda = 0.0;
	idata.lower = tau[past_idx];
	idata.upper = 200;
	double inf_dist = integrate1D(&deSitterLookupKernel, (void*)&lambda, &idata, QAGS);
	idata.lower = tau[future_idx];
	inf_dist += integrate1D(&deSitterLookupKernel, (void*)&lambda, &idata, QAGS);
	if (omega12 > inf_dist)
		return INF;

	//Use upper or lower branch of the solution
	double ml = -1.0 / (cosh(tau[future_idx]) * cosh(tau[future_idx]));
	bool upper_branch = false;

	idata.lower = tau[past_idx];
	idata.upper = tau[future_idx];
	double branch_cutoff = integrate1D(&deSitterLookupKernel, (void*)&ml, &idata, QAGS);
	if (!timelike && omega12 > branch_cutoff) 
		upper_branch = true;

	double x0, mt;
	while (upper - lower > tol && iter < max_iter) {
		x0 = (lower + upper) / 2.0;
		if (!timelike && upper_branch) {
			mt = acosh(1.0 / sqrt(-1.0 * x0));
			idata.lower = tau[past_idx];
			idata.upper = mt;
			res = integrate1D(&deSitterLookupKernel, (void*)&x0, &idata, QAGS);
			assert (res == res);
			idata.lower = tau[future_idx];
			res += integrate1D(&deSitterLookupKernel, (void*)&x0, &idata, QAGS);
			assert (res == res);
			res -= omega12;
			if (res < 0.0)
				lower = x0;
			else
				upper = x0;
		} else {
			idata.lower = tau[past_idx];
			idata.upper = tau[future_idx];
			res = integrate1D(&deSitterLookupKernel, (void*)&x0, &idata, QAGS);
			assert (res == res);
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

	gsl_integration_workspace_free(idata.workspace);

	return distance;
}

//Returns the geodesic distance for two points on a K = 0 de Sitter manifold
inline double distanceDeSitterFlat(Coordinates *c, const float * const tau, const Spacetime &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, int past_idx, int future_idx)
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
	assert (spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("De_Sitter"));
	assert (a > 0.0);
	assert (zeta > HALF_PI);
	assert (zeta1 > HALF_PI);
	assert (zeta > zeta1);
	assert (spacetime.curvatureIs("Flat"));
	assert (past_idx >= 0 && past_idx < N_tar);
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
	bool timelike = nodesAreRelated(c, spacetime, N_tar, a, zeta, zeta1, r_max, alpha, past_idx, future_idx, &omega12);

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

//Returns the embedded distance between two nodes in a 5D embedding
//O(xxx) Efficiency (revise this)
inline double distanceEmb(const float4 &node_a, const float &tau_a, const float4 &node_b, const float &tau_b, const Spacetime &spacetime, const double &a, const double &alpha)
{
	#if DEBUG
	assert (spacetime.stdimIs("4"));
	assert (spacetime.manifoldIs("De_Sitter") || spacetime.manifoldIs("FLRW"));
	assert (spacetime.curvatureIs("Positive"));
	#endif

	//Check if they are the same node
	if (node_a.w == node_b.w && node_a.x == node_b.x && node_a.y == node_b.y && node_a.z == node_b.z)
		return 0.0;

	double alpha_tilde = alpha / a;
	double inner_product_ab;
	double distance;

	double z0_a = 0.0, z0_b = 0.0;
	double z1_a = 0.0, z1_b = 0.0;

	if (spacetime.manifoldIs("FLRW")) {
		IntData idata = IntData();
		idata.tol = 1e-5;

		//Solve for z1 in Rotated Plane
		double power = 2.0 / 3.0;
		z1_a = alpha_tilde * pow(sinh(1.5 * tau_a), power);
		z1_b = alpha_tilde * pow(sinh(1.5 * tau_b), power);

		//Use Numerical Integration for z0
		idata.upper = z1_a;
		z0_a = integrate1D(&embeddedZ1, (void*)&alpha_tilde, &idata, QNG);
		idata.upper = z1_b;
		z1_b = integrate1D(&embeddedZ1, (void*)&alpha_tilde, &idata, QNG);
	} else if (spacetime.manifoldIs("De_Sitter") && spacetime.curvatureIs("Positive")) {
		z0_a = sinh(tau_a);
		z0_b = sinh(tau_b);

		z1_a = cosh(tau_a);
		z1_b = cosh(tau_b);
	}

	if (spacetime.manifoldIs("De_Sitter") && spacetime.curvatureIs("Flat")) {
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

	if (spacetime.manifoldIs("FLRW"))
		inner_product_ab /= POW2(alpha_tilde);

	if (inner_product_ab > 1.0)
		//Timelike
		distance = acosh(inner_product_ab);
	else if (inner_product_ab < -1.0)
		//Disconnected Regions
		//Negative sign indicates not timelike
		distance = -1.0 * INF;
	else
		//Spacelike
		//Negative sign indicates not timelike
		distance = -1.0 * acos(inner_product_ab);

	return distance;
}

//Returns the hyperbolic distance between two nodes in 2D
inline double distanceH(const float2 &hc_a, const float2 &hc_b, const Spacetime &spacetime, const double &zeta)
{
	#if DEBUG
	assert (spacetime.stdimIs("2"));
	assert (spacetime.manifoldIs("Hyperbolic"));
	assert (zeta != 0.0);
	#endif

	if (hc_a.x == hc_b.x && hc_a.y == hc_b.y)
		return 0.0f;

	double distance = acosh(cosh(hc_a.x / zeta) * cosh(hc_b.x / zeta) - sinh(hc_a.x / zeta) * sinh(hc_b.x / zeta) * cos(hc_a.y - hc_b.y)) * zeta;

	return distance;
}

#endif
