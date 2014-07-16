#ifndef CAUSET_H_
#define CAUSET_H_

#include <cstring>
#include <exception>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>

#include <boost/filesystem.hpp>
#include <cuda.h>
#include <curand.h>
#include <GL/freeglut.h>
#include <sys/io.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <fastmath/FastMath.h>
#include <fastmath/FastNumInt.h>
#include <fastmath/ran2.h>
#include <fastmath/stopwatch.h>

#include <printcolor/printcolor.h>

#include "autocorr2.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

#define NPRINT 1000	//Used for debugging statements in loops

#define NBENCH 10	//Times each function is run during benchmarking

#define TOL (1e-28)	//Machine epsilon

#define INF (1e20)	//Machine infinity

#define APPROX false	//Determines whether FastMath approximations are used
			//in computationally intensive routines

#define USE_GSL true	//Use GNU Scientific Library for numerical integration

#define DEBUG false	//Determines whether unit testing is in effect
			//Should be set to false to disable asserts

//////////////////////////////////////////////////////////////////////////////
//References								    //
//[1] Network Cosmology							    //
//    http://www.nature.com/srep/2012/121113/srep00793/full/srep00793.html  //
//[2] Supplementary Information for Network Cosmology			    //
//    http://complex.ffn.ub.es/~mbogunya/archivos_cms/files/srep00793-s1.pdf//
//[3] Uniformly Distributed Random Unit Quaternions			    //
//    mathproofs.blogspot.com/2005/05/uniformly-distributed-random-unit.html//
//////////////////////////////////////////////////////////////////////////////

//Manifold Types
enum Manifold {
	DE_SITTER,
	HYPERBOLIC
};

//Minimal unique properties of a node
struct Node {
	Node() : sc(NULL), tau(NULL), k_in(NULL), k_out(NULL), cc(NULL) {}

	union {
		//Spacetime Coordinates
		float4 *sc;	//Stored as (eta, theta, phi, chi) for DS
		float *theta;	//Spatial Coordinate for 1+1 DS and HYPERBOLIC
	};

	float *tau;	//Rescaled Time for 3+1 DS, Conformal Time for 1+1 DS, and Radius for HYPERBOLIC

	//Number of Neighbors
	int *k_in;
	int *k_out;

	//Connected Component ID
	int *cc;
};

//Sparse edge list vectors
struct Edge {
	Edge() : past_edges(NULL), future_edges(NULL), past_edge_row_start(NULL), future_edge_row_start(NULL) {}

	int *past_edges;		//Sparse adjacency lists
	int *future_edges;
	int *past_edge_row_start;	//Adjacency list indices
	int *future_edge_row_start;
};

//These are conflicts which arise due to over-constraining
//the system with command-line arguments
struct CausetConflicts {
	CausetConflicts() {}

	//Type 0:	a, lambda
	//Type 1:	omegaL, ratio
	//Type 2:	omegaL, tau0
	//Type 3:	tau0, ratio
	//Type 4:	N_tar, delta, alpha, ratio
	//Type 5:	N_tar, delta, alpha, omegaL
	//Type 6:	N_tar, delta, alpha, tau0

	int conflicts[7];
};

//Boolean flags used to reflect command line parameters
struct CausetFlags {
	CausetFlags() : cc(CausetConflicts()), verbose(false), bench(false), yes(false), use_gpu(false), disp_network(false), print_network(false), universe(false), calc_clustering(false), calc_components(false), calc_success_ratio(false), calc_autocorr(false) {}

	CausetConflicts cc;	//Conflicting Parameters

	bool use_gpu;		//Use GPU to Accelerate Select Algorithms
	bool disp_network;	//Plot Network using OpenGL
	bool print_network;	//Print to File
	bool universe;		//Use Universe's Tau Distribution
	
	bool calc_clustering;	//Find Clustering Coefficients
	bool calc_components;	//Find Connected Components
	bool calc_success_ratio;//Find Success Ratio
	bool calc_autocorr;	//Autocorrelation
	
	bool verbose;		//Verbose Output
	bool bench;		//Benchmark Algorithms
	bool yes;		//Suppresses User Input
};

//Numerical parameters constraining the network
struct NetworkProperties {
	NetworkProperties() : N_tar(0), k_tar(0.0), N_res(0), k_res(0.0), N_deg2(0), N_cc(0), N_gcc(0), N_sr(0.0), dim(3), a(1.0), lambda(3.0), zeta(0.0), tau0(0.587582), alpha(0.0), delta(0.0), R0(1.0), omegaM(0.5), omegaL(0.5), ratio(1.0), core_edge_fraction(0.01), edge_buffer(25000), seed(-12345L), graphID(0), flags(CausetFlags()),  manifold(DE_SITTER) {}

	CausetFlags flags;

	int N_tar;			//Target Number of Nodes
	float k_tar;			//Target Average Degree

	int N_res;			//Resulting Number of Connected Nodes
	float k_res;			//Resulting Average Degree

	int N_deg2;			//Nodes of Degree 2 or Greater

	int N_cc;			//Number of Connected Components
	int N_gcc;			//Size of Giant Connected Component

	//Move this to NetworkObservables
	double N_sr;			//Number of Pairs Used in Success Ratio

	int dim;			//Spacetime Dimension (2 or 4)
	Manifold manifold;		//Manifold of the Network

	double a;			//Hyperboloid Pseudoradius
	double lambda;			//Cosmological Constant
	double zeta;			//Pi/2 - Eta_0

	double tau0;			//Rescaled Age of Universe
	double alpha;			//Rescaled Ratio of Matter Density to Dark Energy Density
	double delta;			//Node Density
	double R0;			//Scale Factor at Present Time
	double omegaM;			//Matter Density
	double omegaL;			//Dark Energy Density
	double ratio;			//Ratio of Energy Density to Matter Density

	float core_edge_fraction;	//Fraction of nodes designated as having core edges
	int edge_buffer;		//Small memory buffer for adjacency list

	long seed;			//Random Seed
	int graphID;			//Unique Simulation ID
};

//Measured values of the network
struct NetworkObservables {
	NetworkObservables() : clustering(NULL), average_clustering(0.0), success_ratio(0.0) {}

	float *clustering;		//Clustering Coefficients
	float average_clustering;	//Average Clustering over All Nodes

	float success_ratio;		//Success Ratio
};

//Network object containing minimal unique information
struct Network {
	Network() : network_properties(NetworkProperties()), network_observables(NetworkObservables()), nodes(Node()), edges(Edge()), core_edge_exists(NULL) {}
	Network(NetworkProperties _network_properties) : network_properties(_network_properties), network_observables(NetworkObservables()), nodes(Node()), edges(Edge()), core_edge_exists(NULL) {}

	NetworkProperties network_properties;
	NetworkObservables network_observables;

	Node nodes;
	Edge edges;
	bool *core_edge_exists;		//Adjacency matrix
};

//Algorithmic Performance
struct CausetPerformance {
	CausetPerformance() : sCauset(Stopwatch()), sCalcDegrees(Stopwatch()), sCreateNetwork(Stopwatch()), sGenerateNodes(Stopwatch()), sGenerateNodesGPU(Stopwatch()), sQuicksort(Stopwatch()), sLinkNodes(Stopwatch()), sLinkNodesGPU(Stopwatch()), sMeasureClustering(Stopwatch()), sMeasureConnectedComponents(Stopwatch()), sMeasureSuccessRatio(Stopwatch()) {}

	Stopwatch sCauset;
	Stopwatch sCalcDegrees;
	Stopwatch sCreateNetwork;
	Stopwatch sGenerateNodes;
	Stopwatch sGenerateNodesGPU;
	Stopwatch sQuicksort;
	Stopwatch sLinkNodes;
	Stopwatch sLinkNodesGPU;
	Stopwatch sMeasureClustering;
	Stopwatch sMeasureConnectedComponents;
	Stopwatch sMeasureSuccessRatio;
};

//Benchmark Statistics
struct Benchmark {
	Benchmark() : bCalcDegrees(0.0), bCreateNetwork(0.0), bGenerateNodes(0.0), bGenerateNodesGPU(0.0), bQuicksort(0.0), bLinkNodes(0.0), bLinkNodesGPU(0.0), bMeasureClustering(0.0), bMeasureConnectedComponents(0.0), bMeasureSuccessRatio(0.0) {}

	double bCalcDegrees;
	double bCreateNetwork;
	double bGenerateNodes;
	double bGenerateNodesGPU;
	double bQuicksort;
	double bLinkNodes;
	double bLinkNodesGPU;
	double bMeasureClustering;
	double bMeasureConnectedComponents;
	double bMeasureSuccessRatio;
};

struct GSL_EmbeddedZ1_Parameters {
	double a;
	double alpha;
};

//Custom exception class used in this program
class CausetException : public std::exception
{
	public:
		CausetException() : msg("Unknown Error!") {}
		explicit CausetException(char *_msg) : msg(_msg) {}
		virtual ~CausetException() throw () {}
		virtual const char* what() const throw () { return msg; }

	protected:
		char *msg;
};

//Function prototypes for those described in src/Causet.cu
static NetworkProperties parseArgs(int argc, char **argv);
static bool initializeNetwork(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed);
static bool measureNetworkObservables(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed);
static bool displayNetwork(const Node &nodes, const int * const future_edges, int argc, char **argv);
void display();
static bool loadNetwork(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed);
static bool printNetwork(Network &network, const CausetPerformance &cp, const long &init_seed, const int &gpuID);
static bool printBenchmark(const Benchmark &bm, const CausetFlags &cf);
static void destroyNetwork(Network * const network, size_t &hostMemUsed, size_t &devMemUsed);

#endif
