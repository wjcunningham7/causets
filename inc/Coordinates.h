#ifndef COORDINATES_H_
#define COORDINATES_H_

#include "Causet.h"

/////////////////////////////
//(C) Will Cunningham 2015 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

// 2-D De Sitter Slab
// Spherical Foliation
// Symmetric About eta = 0

// Returns a value for eta
inline float get_2d_sym_sph_deSitter_slab_eta()
{
	return 1.0;
}

#define get_2d_sym_sph_deSitter_slab_theta \
	get_uniform_azimuthal_angle

// Returns an azimuthal uniformly
// distributed in the range [0, 2pi)
inline float get_uniform_azimuthal_angle()
{
	return 1.0;
}

#endif
