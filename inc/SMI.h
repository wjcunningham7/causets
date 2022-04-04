/////////////////////////////
//(C) Will Cunningham 2018 //
//    Perimeter Institute  //
/////////////////////////////

#ifndef SMI_H_
#define SMI_H_

#include "Causet.h"

//This function partitions a causal set by checking if an element is above the specified hypersurface
inline bool smi_hypersurface(Coordinates * const &crd, const Spacetime &spacetime, const double eta0, const double r_max, const int idx)
{
	#if DEBUG
	assert (crd != NULL);
	assert (idx >= 0);
	#endif

	//eta0 is the max. conformal time
	//r_max is the max. radial coordinate

	bool elementAboveHS = false;
	if (spacetime.spacetimeIs("2", "Minkowski", "Slab_N2", "Flat", "None")) {
		if (fabs(crd->y(idx)) < 0.5 - eta0)
			elementAboveHS = true;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Slab_N2", "Flat", "Temporal")) {
		//if (fabs(crd->y(idx)) < 0.5 - eta0)
		if (fabs(crd->y(idx)) < eta_77834_1(2.0 * crd->x(idx), 1.0 - sqrt(1.0 - POW2(2.0 * eta0))) + r_max / 2.0)
			elementAboveHS = true;
	} else if (spacetime.spacetimeIs("2", "Minkowski", "Diamond", "Flat", "None")) {
		double t = crd->x(idx);
		//double x = crd->y(idx);

		if (t > eta0 / 2)	//Upper half of causal set, t > t0/2
			elementAboveHS = true;
	} else if (spacetime.spacetimeIs("4", "De_Sitter", "Slab", "Positive", "None")) {
		double t = crd->w(idx);
		double theta1 = crd->x(idx);
		double theta2 = crd->y(idx);
		double theta3 = crd->z(idx);

		if (t > theta1 + theta2 + theta3)
			elementAboveHS = true;
	} else {
		//throw CausetException("Spacetime not supported! Find this error in <inc/SMI.h>\n");
		return false;
	}

	return elementAboveHS;
}

#endif
