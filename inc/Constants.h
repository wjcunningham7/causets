#ifndef CONSTANTS_H_
#define CONSTANTS_H_

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
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
			//If this is false, in some places series approximations will be used

#define EMBED_NODES false	//Embed spatial coordinates in a higher dimension
				//For flat space this means use Cartesian coordinates (not spherical)

#define DIST_V2 true	//Use factored (v2) or expanded (v1) distance formulae
			//Version 2 uses fewer operations, and is faster

#define SPECIAL_SAUCER false	//The special saucer has surfaces which intersect at right angles

#define VE_RANDOM true	//Pick random pairs when performing embedding validation test

#define VD_RANDOM true	//Pick random pairs when performing distance validation test

#define SR_RANDOM true	//Pick random pairs when calculating success ratio

#define VP_RANDOM true	//Pick random pairs when calculating vector products

#define TRAVERSE_V2 false	//Version 2 (and 3) uses spatial distances rather than geodesics

#define TRAVERSE_VECPROD false	//Use inner products of the embedded nodes

#define ACTION_V2 true		//Version 2 uses OpenMP, Version 3 uses minimal bitwise optimizations
				//If false, version 1 is used, with no parallelization
				//If MPI is enabled, this flag is ignored

#define ACTION_MPI_V5 262144	//Size limit which determines which MPI algorithm is used

#define GEODESIC_LOWER 0.95	//Boundary between Region I and Region II approximations
				//Should be less than 1.0

#define GEODESIC_UPPER 1.05	//Boundary between Region II and Region III approximations
				//Should be greater than 1.0

//Debugging Flags
#define DEBUG true	//Determines whether unit testing is in effect
			//Set to false to disable assert statements
			//In general it should be set to true since there
			//is a minimal performance difference

#define NPRINT 1000	//Used for debugging statements inside loops

//Benchmarking Flags
#define NBENCH 10	//Number of samples used during benchmarking

#ifdef CUDA_ENABLED

//CUDA Flags
#define GPU_MIN 384	//Minimum number of nodes needed for GPU linking
			//Maybe this can be changed, but it probably shouldn't be...

#define BLOCK_SIZE 128	//Number of threads per block
			//This value is dependent on the GPU architecture
			//DO NOT EDIT THIS VALUE

#define THREAD_SIZE 4	//Number of element operations per thread
			//This value is dependent on the GPU shared memory cache size
			//DO NOT EDIT THIS VALUE

#define NBUFFERS 4	//Number of memory buffers used concurrently on GPU
			//This value is dependent on the GPU global memory cache size
			//DO NOT EDIT THIS VALUE

//Options for GPU Algorithms
//See src/NetworkCreator_GPU.cu
#define LINK_NODES_GPU_V2 true		//Version 2 finds multiple links per thread
					//It can also handle larger graphs than version 1
					//There's no reason to set this to false

#define GEN_ADJ_LISTS_GPU_V2 true	//Version 2 performs fewer operations and has
					//been optimized to a higher degree
					//It also supports MPI and asyncrhonous operations
					//This can safely be set to false if MPI is not used

#define DECODE_LISTS_GPU_V2 true	//Deprecated mostly since decoding is done on the CPU
					//Version 2 can handle larger graphs

#endif

#endif
