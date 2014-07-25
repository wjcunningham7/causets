#ifndef CONSTANTS_H_
#define CONSTANTS_H_

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

//Numerical Constants
#define TOL (1e-28)	//Zero

#define INF (1e20)	//Infinity

#define G (6.67384e-11)	//Gravitational Constant

//Algorithmic Flags
#define APPROX false	//Determines whether FastMath approximations are used
			//in computationally intensive subroutines

#define USE_GSL true	//Use GNU Scientific Library for numerical integration

//Debugging Flags
#define DEBUG false	//Determines whether unit testing is in effect
			//Set to false to disable assert statements

#define NPRINT 1000	//Used for debugging statements inside loops

//Benchmarking Flags
#define NBENCH 10	//Number of samples used during benchmarking

#endif
