#ifndef CAUSET_H_
#define CAUSET_H_

#include <cstring>
#include <exception>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>

#include <cuda.h>
#include <curand.h>
#include <GL/freeglut.h>

#include "autocorr2.h"
#include "FastMath.h"
#include "ran2.h"
#include "stopwatch.h"

//#define TOL (1e-28)	//Any value smaller than this is rounded to zero
#define NPRINT 10000	//Used for debugging statements in loops
#define NBENCH 10	//Times each function is run during benchmarking

////////////////////////////////////////////////////////////////////////////
//References
//[1] Network Cosmology
//    http://www.nature.com/srep/2012/121113/srep00793/full/srep00793.html
//[2] Supplementary Information for Network Cosmology
//    http://complex.ffn.ub.es/~mbogunya/archivos_cms/files/srep00793-s1.pdf
//[3] Uniformly Distributed Random Unit Quaternions
//    mathproofs.blogspot.com/2005/05/uniformly-distributed-random-unit.html
////////////////////////////////////////////////////////////////////////////

//Manifold Types
//Currently only DE_SITTER is supported
enum Manifold {
	EUCLIDEAN,
	DE_SITTER,
	ANTI_DE_SITTER
};

//Minimal unique properties of a node
struct Node {
	Node() : t(0.0), theta(0.0), phi(0.0), chi(0.0), k_in(0), k_out(0) {}
	Node(float _t, float _theta, float _phi, float _chi, int _k_in, int _k_out) : t(_t), theta(_theta), phi(_phi), chi(_chi), k_in(_k_in), k_out(_k_out) {}

	//Temporal Coordinate
	float t;	//Note this is 't' for 1+1 and 3+1, and 'tau' for universe

	//Spatial Coordinates
	float theta;
	float phi;
	float chi;

	//Number of Neighbors
	int k_in;
	int k_out;
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
	CausetFlags() : cc(CausetConflicts()), verbose(false), bench(false), use_gpu(false), disp_network(false), print_network(false), universe(false), calc_clustering(false), calc_autocorr(false) {}
	CausetFlags(CausetConflicts _cc, bool _verbose, bool _bench, bool _use_gpu, bool _disp_network, bool _print_network, bool _universe, bool _calc_clustering, bool _calc_autocorr) : cc(_cc), verbose(_verbose), bench(_bench), use_gpu(_use_gpu), disp_network(_disp_network), print_network(_print_network), universe(_universe), calc_clustering(_calc_clustering), calc_autocorr(_calc_autocorr) {}

	CausetConflicts cc;	//Conflicting Parameters

	bool use_gpu;		//Use GPU to Accelerate Select Algorithms
	bool disp_network;	//Plot Network using OpenGL
	bool print_network;	//Print to File
	bool universe;		//Use Universe's Tau Distribution
	
	bool calc_clustering;	//Find Clustering Coefficients
	bool calc_autocorr;	//Autocorrelation
	
	bool verbose;		//Verbose Output
	bool bench;		//Benchmark Algorithms
};

//CUDA Kernel Execution Parameters
struct NetworkExec {
	NetworkExec() : threads_per_block(dim3(256, 256, 1)), blocks_per_grid(dim3(256, 256, 1)) {}
	NetworkExec(dim3 tpb, dim3 bpg) : threads_per_block(tpb), blocks_per_grid(bpg) {}

	dim3 threads_per_block;
	dim3 blocks_per_grid;
};

//Numerical parameters constraining the network
struct NetworkProperties {
	NetworkProperties() : N_tar(0), k_tar(0.0), N_res(0), k_res(0.0), N_deg2(0), dim(3), a(1.0), lambda(3.0), zeta(0.0), tau0(0.587582), alpha(0.0), delta(1.0), R0(1.0), omegaM(0.5), omegaL(0.5), ratio(1.0), core_edge_fraction(0.01), edge_buffer(25000), seed(-12345L), graphID(0), flags(CausetFlags()), network_exec(NetworkExec()), manifold(DE_SITTER) {}
	NetworkProperties(int _N_tar, float _k_tar, int _dim, double _a, double _lambda, double _zeta, double _tau0, double _alpha, double _delta, double _R0, double _omegaM, double _omegaL, double _ratio, float _core_edge_fraction, int _edge_buffer, long _seed, int _graphID, CausetFlags _flags, NetworkExec _network_exec, Manifold _manifold) : N_tar(_N_tar), k_tar(_k_tar), N_res(0), k_res(0), N_deg2(0), dim(_dim), a(_a), lambda(_lambda), zeta(_zeta), tau0(_tau0), alpha(_alpha), delta(_delta), R0(_R0), omegaM(_omegaM), omegaL(_omegaL), ratio(_ratio), core_edge_fraction(_core_edge_fraction), edge_buffer(_edge_buffer), seed(_seed), graphID(_graphID), flags(_flags), network_exec(_network_exec), manifold(_manifold) {}

	CausetFlags flags;
	NetworkExec network_exec;

	int N_tar;			//Target Number of Nodes
	float k_tar;			//Target Average Degree

	int N_res;			//Resulting Number of Connected Nodes
	float k_res;			//Resulting Average Degree

	int N_deg2;			//Nodes of Degree 2 or Greater

	int dim;			//Spacetime Dimension (2 or 4)
	Manifold manifold;		//Manifold of the Network

	double a;			//Hyperboloid Pseudoradius
	double lambda;			//Cosmological Constant
	double zeta;			//Pi/2 - Eta_0
					//Note Eta_0 is stored here for 1+1

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
	NetworkObservables() : clustering(NULL), average_clustering(0.0) {}
	NetworkObservables(float *_clustering, float _average_clustering) : clustering(_clustering), average_clustering(_average_clustering) {}

	float *clustering;		//Clustering Coefficients
	float average_clustering;	//Average Clustering over All Nodes
};

//Network object containing minimal unique information
struct Network {
	Network() : network_properties(NetworkProperties()), network_observables(NetworkObservables()), nodes(NULL), past_edges(NULL), future_edges(NULL), past_edge_row_start(NULL), future_edge_row_start(NULL), core_edge_exists(NULL) {}
	Network(NetworkProperties _network_properties) : network_properties(_network_properties), network_observables(NetworkObservables()), nodes(NULL), past_edges(NULL), future_edges(NULL), past_edge_row_start(NULL), future_edge_row_start(NULL), core_edge_exists(NULL) {}
	Network(NetworkProperties _network_properties, NetworkObservables _network_observables, Node *_nodes, int *_past_edges, int *_future_edges, int *_past_edge_row_start, int *_future_edge_row_start, bool *_core_edge_exists) : network_properties(_network_properties), network_observables(_network_observables), nodes(_nodes), past_edges(_past_edges), future_edges(_future_edges), past_edge_row_start(_past_edge_row_start), future_edge_row_start(_future_edge_row_start), core_edge_exists(_core_edge_exists) {}

	NetworkProperties network_properties;
	NetworkObservables network_observables;

	Node *nodes;
	int *past_edges;		//Sparse adjacency lists
	int *future_edges;
	int *past_edge_row_start;	//Adjacency list indices
	int *future_edge_row_start;
	bool *core_edge_exists;		//Adjacency matrix

	//GPU Memory Pointers
	CUdeviceptr d_nodes;
	CUdeviceptr d_edges;
};

//Algorithmic Performance
struct CausetPerformance {
	CausetPerformance() : sCauset(Stopwatch()), sCreateNetwork(Stopwatch()), sGenerateNodes(Stopwatch()), sQuicksort(Stopwatch()), sLinkNodes(Stopwatch()), sMeasureClustering(Stopwatch()) {}

	Stopwatch sCauset;
	Stopwatch sCreateNetwork;
	Stopwatch sGenerateNodes;
	Stopwatch sQuicksort;
	Stopwatch sLinkNodes;
	Stopwatch sMeasureClustering;
};

//Benchmark Statistics
struct Benchmark {
	Benchmark() : bCreateNetwork(0.0), bGenerateNodes(0.0), bQuicksort(0.0), bLinkNodes(0.0), bMeasureClustering(0.0) {}

	double bCreateNetwork;
	double bGenerateNodes;
	double bQuicksort;
	double bLinkNodes;
	double bMeasureClustering;
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
NetworkProperties parseArgs(int argc, char **argv);
bool initializeNetwork(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed);
bool measureNetworkObservables(Network * const network, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed);
bool displayNetwork(const Node * const nodes, const int * const future_edges, int argc, char **argv);
void display();
bool loadNetwork(Network * const network, CausetPerformance * const cp, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed);
bool printNetwork(Network &network, const CausetPerformance &cp, const long &init_seed, const int &gpuID);
bool printBenchmark(const Benchmark &bm, const CausetFlags &cf);
void destroyNetwork(Network * const network, size_t &hostMemUsed, size_t &devMemUsed);

#endif
