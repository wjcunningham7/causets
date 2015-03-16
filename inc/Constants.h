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

#define MTAU (3e6)	//Used to identify tau approximations

#define G (6.67384e-11)	//Gravitational Constant

//Algorithmic Flags
#define APPROX false	//Determines whether FastMath approximations are used
			//in computationally intensive subroutines

#define USE_GSL true	//Use GNU Scientific Library for numerical integration

#define DIST_V2 true	//Use factored (v2) or expanded (v1) distance formulae

//Debugging Flags
#define DEBUG true	//Determines whether unit testing is in effect
			//Set to false to disable assert statements

#define NPRINT 1000	//Used for debugging statements inside loops

//Benchmarking Flags
#define NBENCH 10	//Number of samples used during benchmarking

#ifdef CUDA_ENABLED

//CUDA Flags
#define BLOCK_SIZE 128	//Number of threads per block

#define GROUP_SIZE 8	//Number of block groups per grid dimension
			//Increase this by a power of 2 if too much GPU memory is
			//requested in the generateLists() algorithm

#define THREAD_SIZE 4	//Number of element operations per thread

#define NBUFFERS 4	//Number of memory buffers used concurrently on GPU

//Options for GPU Algorithms
//See src/NetworkCreator_GPU.cu
#define LINK_NODES_GPU_V2 true

#define GEN_ADJ_LISTS_GPU_V2 true

#define DECODE_LISTS_GPU_V2 true

#endif

#endif
