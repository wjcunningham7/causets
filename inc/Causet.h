/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef CAUSET_H_
#define CAUSET_H_

/*#ifdef __CUDACC_VER__
#undef __CUDACC_VER__
#define __CUDACC_VER__ 90000
#endif*/

#ifdef AVX2_ENABLED
#include <x86intrin.h>
#endif

//Core System Files
#include <cstring>
#include <exception>
#include <fstream>
#include <getopt.h>
#define __STDC_FORMAT_MACROS	//Required to print uint64_t variables
#include <inttypes.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <math.h>
#include <numeric>
#include <pthread.h>
#include <sstream>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

//System Files for Parallel Acceleration
#ifdef CUDA_ENABLED
  #include <cuda.h>
  #include <curand_kernel.h>
#endif

#ifdef MPI_ENABLED
  #include <mpi.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num() 0
  #define omp_get_max_threads() 1
#endif

//Other System Files
#include <boost/functional/hash/hash.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/unordered_map.hpp>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_sf_lambert.h>
#include <gsl/gsl_statistics_double.h>
#include <sys/io.h>

//Custom System Files
#include "config.h" //Required before FastBitset.h
#include <fastmath/fastbitset.h>
#include <fastmath/fastmath.h>
#include <fastmath/integration.h>
#include <fastmath/stopwatch.h>
#include <fastmath/printcolor.h>
#include <fastmath/progressbar.h>
#include <fastmath/resources.h>

//Local Files
#include "Constants.h"
#include "Spacetime.h"

using namespace fastmath;

//////////////////////////////////////////////////////////////////////////////
//References								    //
//[1] Network Cosmology							    //
//    http://www.nature.com/srep/2012/121113/srep00793/full/srep00793.html  //
//[2] Supplementary Information for Network Cosmology			    //
//    http://complex.ffn.ub.es/~mbogunya/archivos_cms/files/srep00793-s1.pdf//
//[3] Uniformly Distributed Random Unit Quaternions			    //
//    mathproofs.blogspot.com/2005/05/uniformly-distributed-random-unit.html//
//[4] Approximations for Elliptic Integrals				    //
//    www.jstor.org/stable/2004539					    //
//////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_ENABLED
//Redefine CUDA data types

struct __attribute__ ((aligned(8))) float2 {
	float x, y;
};

struct __attribute__ ((aligned(16))) double2 {
	double x, y;
};

extern inline float2 make_float2(float x, float y)
{
	float2 f;
	f.x = x;
	f.y = y;
	return f;
}

struct float3 {
	float x, y, z;
};

extern inline float3 make_float3(float x, float y, float z)
{
	float3 f;
	f.x = x;
	f.y = y;
	f.z = z;
	return f;
}

struct __attribute__ ((aligned(16))) float4 {
	float w, x, y, z;
};

extern inline float4 make_float4(float w, float x, float y, float z)
{
	float4 f;
	f.w = w;
	f.x = x;
	f.y = y;
	f.z = z;
	return f;
}

struct float5 {
	float v, w, x, y, z;
};

typedef int CUcontext;

#else

struct float5 {
	float v, w, x, y, z;
};

#endif

extern inline float5 make_float5(float v, float w, float x, float y, float z)
{
	float5 f;
	f.v = v;
	f.w = w;
	f.x = x;
	f.y = y;
	f.z = z;
	return f;
}

//Boost RNG
typedef boost::mt19937 Engine;
typedef boost::uniform_real<double> UDistribution;
typedef boost::normal_distribution<double> NDistribution;
typedef boost::poisson_distribution<> PDistribution;
typedef boost::variate_generator<Engine&, UDistribution> UGenerator;
typedef boost::variate_generator<Engine&, NDistribution> NGenerator;
typedef boost::variate_generator<Engine&, PDistribution> PGenerator;

struct MersenneRNG {
	MersenneRNG() : dist(0.0, 1.0), rng(eng, dist)  {}

	Engine eng;
	UDistribution dist;
	UGenerator rng;
};

//Causal Set Resources
struct CaResources {
	CaResources() : hostMemUsed(0), maxHostMemUsed(0), devMemUsed(0), maxDevMemUsed(0) {}

	//Memory Allocated (in bytes)
	size_t hostMemUsed;
	size_t maxHostMemUsed;
	size_t devMemUsed;
	size_t maxDevMemUsed;
};

//These coordinate data structures are important because they allow
//the data to be coalesced (physically adjacent), and this can improve the
//speed of global memory reads on the GPU by a factor of 8 or 16.  Further,
//it provides a way for higher-dimensional nodes to be added easily at
//a later date.

//Abstract N-Dimensional Vertex Coordinate
//This should not be instantiated by itself
struct Coordinates {
	Coordinates(int _ndim) : ndim(_ndim), zero(0.0), null_ptr(NULL) {
		points = new float*[_ndim];
	}
	virtual ~Coordinates() { delete [] this->points; this->points = NULL; }

	int getDim() { return ndim; }
	bool isNull() { return points == NULL; }

	//These functions should not be accessed through this structure.  They
	//should be accessed through inherited structures to avoid SLICING.
	//These virtual definitions are used to indicate bugs in the code,
	//usually when a structure is passed by value instead of by reference

	virtual float & v(unsigned int idx) { return zero; }
	virtual float & w(unsigned int idx) { return zero; }
	virtual float & x(unsigned int idx) { return zero; }
	virtual float & y(unsigned int idx) { return zero; }
	virtual float & z(unsigned int idx) { return zero; }

	virtual float *& v(void) { return null_ptr; }
	virtual float *& w(void) { return null_ptr; }
	virtual float *& x(void) { return null_ptr; }
	virtual float *& y(void) { return null_ptr; }
	virtual float *& z(void) { return null_ptr; }

	virtual float2 getFloat2(unsigned int idx) { return make_float2(0.0f, 0.0f); }
	virtual void setFloat2(float2 val, unsigned int idx) {}
	virtual float3 getFloat3(unsigned int idx) { return make_float3(0.0f, 0.0f, 0.0f); }
	virtual void setFloat3(float3 val, unsigned int idx) {}
	virtual float4 getFloat4(unsigned int idx) { return make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
	virtual void setFloat4(float4 val, unsigned int idx) {}
	virtual float5 getFloat5(unsigned int idx) { return make_float5(0.0f, 0.0f, 0.0f, 0.0f, 0.0f); }
	virtual void setFloat5(float5 val, unsigned int idx) {}

protected:
	float **points;

private:
	int ndim;
	float zero;
	float *null_ptr;
};

//2-Dimensional Vertex Coordinate
struct Coordinates2D : Coordinates {
	Coordinates2D() : Coordinates(2) {
		this->points[0] = NULL;
		this->points[1] = NULL;
	}

	Coordinates2D(float *_x, float *_y) : Coordinates(2) {
		this->points[0] = _x;
		this->points[1] = _y;
	}

	//Access and mutate element as c.x(index)
	float & x(unsigned int idx) { return this->points[0][idx]; }
	float & y(unsigned int idx) { return this->points[1][idx]; }

	//Access pointer as c.x()
	float *& x() { return this->points[0]; }
	float *& y() { return this->points[1]; }

	//Return compressed value
	float2 getFloat2(unsigned int idx) {
		float2 f;
		f.x = this->points[0][idx];
		f.y = this->points[1][idx];
		return f;
	}

	void setFloat2(float2 val, unsigned int idx) {
		this->points[0][idx] = val.x;
		this->points[1][idx] = val.y;
	}

	//Access pointer as c[0]
	//Keeping this for my own reference but don't use this style for now	
	//float * operator [] (unsigned int i) const { return this->points[i]; }
	//float *& operator [] (unsigned int i) { return this->points[i]; }
};

//3-Dimensional Vertex Coordinate
struct Coordinates3D : Coordinates {
	Coordinates3D() : Coordinates(3) {
		this->points[0] = NULL;
		this->points[1] = NULL;
		this->points[2] = NULL;
	}

	float & x(unsigned int idx) { return this->points[0][idx]; }
	float & y(unsigned int idx) { return this->points[1][idx]; }
	float & z(unsigned int idx) { return this->points[2][idx]; }

	float *& x() { return this->points[0]; }
	float *& y() { return this->points[1]; }
	float *& z() { return this->points[2]; }

	float3 getFloat3(unsigned int idx) {
		float3 f;
		f.x = this->points[0][idx];
		f.y = this->points[1][idx];
		f.z = this->points[2][idx];
		return f;
	}

	void setFloat3(float3 val, unsigned int idx) {
		this->points[0][idx] = val.x;
		this->points[1][idx] = val.y;
		this->points[2][idx] = val.z;
	}
};

//4-Dimensional Vertex Coordinate
struct Coordinates4D : Coordinates {
	Coordinates4D() : Coordinates(4) {
		this->points[0] = NULL;
		this->points[1] = NULL;
		this->points[2] = NULL;
		this->points[3] = NULL;
	}

	Coordinates4D(float *& _w, float *& _x, float *& _y, float *& _z) : Coordinates(4) {
		this->points[0] = _w;
		this->points[1] = _x;
		this->points[2] = _y;
		this->points[3] = _z;
	}

	float & w(unsigned int idx) { return this->points[0][idx]; }
	float & x(unsigned int idx) { return this->points[1][idx]; }
	float & y(unsigned int idx) { return this->points[2][idx]; }
	float & z(unsigned int idx) { return this->points[3][idx]; }

	float *& w() { return this->points[0]; }
	float *& x() { return this->points[1]; }
	float *& y() { return this->points[2]; }
	float *& z() { return this->points[3]; }

	float4 getFloat4(unsigned int idx) {
		float4 f;
		f.w = this->points[0][idx];
		f.x = this->points[1][idx];
		f.y = this->points[2][idx];
		f.z = this->points[3][idx];
		return f;
	}

	void setFloat4(float4 val, unsigned int idx) {
		this->points[0][idx] = val.w;
		this->points[1][idx] = val.x;
		this->points[2][idx] = val.y;
		this->points[3][idx] = val.z;
	}
};

//5-Dimensional Vertex Coordinate
//Typically used for embedded 4D
struct Coordinates5D : Coordinates {
	Coordinates5D() : Coordinates(5) {
		this->points[0] = NULL;
		this->points[1] = NULL;
		this->points[2] = NULL;
		this->points[3] = NULL;
		this->points[4] = NULL;
	}

	Coordinates5D(float *_v, float *_w, float *_x, float *_y, float *_z) : Coordinates(5) {
		this->points[0] = _v;
		this->points[1] = _w;
		this->points[2] = _x;
		this->points[3] = _y;
		this->points[4] = _z;
	}

	float & v(unsigned int idx) { return this->points[0][idx]; }
	float & w(unsigned int idx) { return this->points[1][idx]; }
	float & x(unsigned int idx) { return this->points[2][idx]; }
	float & y(unsigned int idx) { return this->points[3][idx]; }
	float & z(unsigned int idx) { return this->points[4][idx]; }

	float *& v() { return this->points[0]; }
	float *& w() { return this->points[1]; }
	float *& x() { return this->points[2]; }
	float *& y() { return this->points[3]; }
	float *& z() { return this->points[4]; }

	float5 getFloat5(unsigned int idx) {
		float5 f;
		f.v = this->points[0][idx];
		f.w = this->points[1][idx];
		f.x = this->points[2][idx];
		f.y = this->points[3][idx];
		f.z = this->points[4][idx];
		return f;
	}

	void setFloat5(float5 val, unsigned int idx) {
		this->points[0][idx] = val.v;
		this->points[1][idx] = val.w;
		this->points[2][idx] = val.x;
		this->points[3][idx] = val.y;
		this->points[4][idx] = val.z;
	}
};

enum GraphType {
	RGG = 0,
	KR_ORDER = 1,
	_2D_ORDER = 2,
	LATTICE = 3,
	RANDOM = 4,
	RANDOM_DAG = 5,
	ANTICHAIN = 6
};

static constexpr const char *gt_strings[] = { "RGG", "KR_ORDER", "2D_ORDER", "LATTICE", "RANDOM", "RANDOM_DAG", "ANTICHAIN" };

//Function used in the Metropolis step
//of Monte Carlo evolutions of the causal set
enum WeightFunction {
	WEIGHTLESS = 0,
	BD_ACTION_2D = 1,
	BD_ACTION_3D = 2,
	BD_ACTION_4D = 3,
	RELATION = 4,
	RELATION_PAIR = 5,
	LINK_PAIR = 6,
	DIAMOND = 7,
	ANTIPERCOLATION = 8
};

static constexpr const char *wf_strings[] = { "WEIGHTLESS", "BD_ACTION_2D", "BD_ACTION_3D", "BD_ACTION_4D", "RELATION", "RELATION_PAIR", "LINK_PAIR", "DIAMOND", "ANTIPERCOLATION" };

//Node ID
//This is a bit of a messy hack to deal with extra variables
//in both causal sets and hyperbolic models which don't fit elsewhere.
//So the 'id' of a causal set node is said to be its rescaled time,
//and the 'id' of a hyperbolic node is its AS identification number.
union ID {
	ID() { memset(this, 0, sizeof(ID)); }

	float *tau;			//Rescaled Time for DS
	int *AS;			//Autonomous System (AS) ID number for HYPERBOLIC
};

//Minimal unique properties of a node
struct Node {
	Node() : crd(NULL), id(ID()), k_in(NULL), k_out(NULL), cc_id(NULL) {}

	//Node Identifiers
	Coordinates *crd;	//Assign a derived type to this.  This is a generalized
				//method of creating an N-dimensional node.
	ID id;

	//HashMap for HYPERBOLIC
	boost::unordered_map<int, int> AS_idx;

	//Number of Neighbors
	int *k_in;	//In-Degrees
	int *k_out;	//Out-Degrees

	//Connected Component ID
	int *cc_id;
};

//Sparse edge list vectors
struct Edge {
	Edge() : past_edges(NULL), future_edges(NULL), past_edge_row_start(NULL), future_edge_row_start(NULL) {}

	int *past_edges;		//Sparse adjacency lists
	int *future_edges;
	int64_t *past_edge_row_start;	//Adjacency list indices
	int64_t *future_edge_row_start;
};

enum CausetSpinlock {
	UNLOCKED,
	LOCKED
};

enum MPISignal {
	REQUEST_UNLOCK,
	REQUEST_LOCK,
	REQUEST_UPDATE_AVAIL,
	REQUEST_UPDATE_NEW,
	REQUEST_EXCHANGE
};

struct CausetMPI {
	CausetMPI() : lock(UNLOCKED), num_mpi_threads(1), rank(0), fail(0) {}

	Bitvector adj_buf;		//Buffer used for adjacency matrix memory swaps
	CausetSpinlock lock;		//Spinlock for shared resources

	int num_mpi_threads;		//Number of MPI Threads
	int rank;			//ID of this MPI Thread
	int fail;			//Flag used to tell all nodes to return
};

//Boolean flags used to reflect command line parameters
struct CausetFlags {
	CausetFlags() : use_gpu(false), decode_cpu(false), print_network(false), print_edges(false), print_dot(false), print_hdf5(false), growing(false), link(false), relink(false), link_epso(false), mcmc(false), exchange_replicas(false), cluster_flips(false), popanneal(false), has_exact_k(false), binomial(false), read_old_format(false), quiet_read(false), no_pos(false), use_bit(true), mpi_split(false), calc_clustering(false), calc_components(false), calc_success_ratio(false), calc_stretch(false), calc_action(false), calc_action_theory(false), calc_chain(false), calc_hubs(false), calc_geo_dis(false), calc_antichain(false), calc_entanglement_entropy(false), calc_foliation(false), calc_dimension(false), calc_smi(false), calc_extrinsic_curvature(false), strict_routing(false), verbose(false), bench(false), yes(false), test(false) {}

	bool use_gpu;			//Use GPU to Accelerate Select Algorithms
	bool decode_cpu;		//Decode edge list using serial sort
	bool print_network;		//Print to File
	bool print_edges;		//Print Edge List to File
	bool print_dot;			//Print Edges to Dot File
	bool print_hdf5;		//Print Using HDF5 Format
	bool growing;			//Use Static/Growing H2 Model
	bool link;			//Link Nodes after Generation
	bool relink;			//Link Nodes in Graph Identified by 'graphID'
	bool link_epso;			//Link Nodes in H2 Using EPSO Rule
	bool mcmc;			//Markov Chain Monte Carlo (with Metropolis update)
	bool exchange_replicas;		//Replica Exchange Monte Carlo
	bool cluster_flips;		//Attempt Wolff cluster flips
	bool popanneal;			//Evolve using Population Annealing
	bool has_exact_k;		//True if there exists an exact expression for <k>
	bool binomial;			//Use exactly N_tar elements; otherwise choose a Poisson
					//random variable with mean N_tar

	bool read_old_format;		//Read Node Positions in the Format (theta3, theta2, theta1)
	bool quiet_read;		//Ignore Warnings when Reading Graph
	bool no_pos;			//No positions in graph (edge list only)
	bool use_bit;			//Use bit array instead of sparse edge lists
	bool mpi_split;			//When MPI is enabled, split the adjacency matrix
					//among all computers
	
	bool calc_clustering;		//Find Clustering Coefficients
	bool calc_components;		//Find Connected Components
	bool calc_success_ratio;	//Find Success Ratio
	bool calc_stretch;		//Measure Stretch
	bool calc_action;		//Measure Action
	bool calc_action_theory;	//Calculate Theoretical Action
	bool calc_chain;		//Study Maximum Chain Lengths
	bool calc_hubs;			//Calculate Hub Connectivity
	bool calc_geo_dis;		//Calculate Fraction of Geodesically Disconnected Pairs
	bool calc_antichain;		//Identify a Maximal Random Antichain
	bool calc_entanglement_entropy;	//Calculate the Entanglement Entropy
	bool calc_foliation;		//Generate a spacetime foliation
	bool calc_dimension;		//Estimate the spacetime dimension
	bool calc_smi;			//Calculate the spacetime mutual information
	bool calc_extrinsic_curvature;	//Calculate extrinsic curvature of boundaries

	bool strict_routing;		//Use Strict Routing Protocol (see notes)
	
	bool verbose;			//Verbose Output
	bool bench;			//Benchmark Algorithms
	bool yes;			//Suppresses User Input
	bool test;			//Test Parameters
};

//Numerical parameters constraining the network
struct NetworkProperties {
	NetworkProperties() : flags(CausetFlags()), spacetime(Spacetime()), N_tar(0), k_tar(0.0), N(0), sweeps(0), cluster_rate(0.0), N_sr(0.0), N_hubs(0), N_gd(0.0), entropy_size(0.0), a(0.0), eta0(0.0), zeta(0.0), zeta1(0.0), r_max(0.0), tau0(0.0), alpha(0.0), delta(0.0), K(0.0), beta(0.0), epsilon(0.0), mu(0.0), gamma(0.0), omegaM(0.0), omegaL(0.0), core_edge_fraction(1.0), edge_buffer(0.0), gt(RGG), wf(WEIGHTLESS), seed(12345L), graphID(0), cmpi(CausetMPI()), mrng(MersenneRNG()), group_size(1), datdir("./data/"), R0(128), runs(1) {}

	CausetFlags flags;
	Spacetime spacetime;		//Encodes dimension, manifold, region, curvature, and symmetry

	int N_tar;			//Target Number of Nodes
	float k_tar;			//Target Average Degree
	int N;				//Actual Number of Nodes

	unsigned sweeps;		//Number of sweeps used for MMC
	float cluster_rate;		//Rate at which cluster flips are attempted
	double couplings[2] = {0,0};	//Couplings used for generalized action

	long double N_sr;		//Number of Pairs Used in Success Ratio
	int N_actth;			//2^(N_actth) is the Largest Theoretical Value to Calculate
	int N_hubs;			//Number of Nodes Used to Calculate Hub Connectivity
	long double N_gd;		//Number of Pairs Used in Geodesic Disconnectedness Measurements
	float entropy_size;		//Fraction of causal interval height the inner interval takes

	double a;			//Hyperboloid Pseudoradius
	double eta0;			//Maximum Conformal Time
	double zeta;			//Pi/2 - Eta_0
	double zeta1;			//Pi/2 - Eta_1

	double r_max;			//Size of the Spatial Slice (Radius)
	double tau0;			//Rescaled Age
	double alpha;			//Rescaled Ratio of Matter Density to Dark Energy Density
	double delta;			//Node Density
	double K;			//Extrinsic curvature

	double beta;			//Inverse Temperature
	double epsilon;			//Action Smearing Parameter
	double mu;			//Chemical Potential
	double gamma;			//Degree Exponent

	double omegaM;			//Matter Density
	double omegaL;			//Dark Energy Density

	float core_edge_fraction;	//Fraction of nodes designated as having core edges
	float edge_buffer;		//Fraction of edge list added as a buffer

	GraphType gt;			//Type of graph
	WeightFunction wf;		//Weight function used in MMC
	long seed;			//Random Seed
	int graphID;			//Unique Simulation ID

	CausetMPI cmpi;			//MPI Flags

	MersenneRNG mrng;		//Mersenne Twister RNG

	int group_size;			//Number of mega-blocks per grid dimension

	std::string datdir;		//Directory to read and write data

	//Population Annealing
	unsigned R0;			//Target Population Size
	unsigned runs;			//Number of Sequential Runs
};

//Measured values of the network
struct NetworkObservables {
	NetworkObservables() : N_res(0), k_res(0.0f), N_deg2(0), N_cc(0), N_gcc(0), clustering(NULL), average_clustering(0.0), success_ratio(0.0), success_ratio2(0.0), stretch(0.0), cardinalities(NULL), action(0.0f), chaintime(NULL), actth(NULL), longest_chain_length(0), hub_density(0.0), hub_densities(NULL), geo_discon(0.0), dimension(0.0), smi(NULL), layers(NULL), extrinsic_curvature(0.0) {}
	
	int N_res;			//Resulting Number of Connected Nodes
	float k_res;			//Resulting Average Degree

	int N_deg2;			//Nodes of Degree 2 or Greater

	int N_cc;			//Number of Connected Components
	int N_gcc;			//Size of Giant Connected Component

	float *clustering;		//Clustering Coefficients
	float average_clustering;	//Average Clustering over All Nodes

	float success_ratio;		//Success Ratio (type 1 - all pairs which are geodesically connected are considered)
	float success_ratio2;		//Success Ratio (type 2 - pairs which are geodesically disconnected, but have neighbors which are connected, are considered here)
	float stretch;			//Average Stretch Across Greedy Paths

	uint64_t *cardinalities;	//M-Element Inclusive-Order-Interval Cardinalities
	double action;			//Action

	std::vector<unsigned int> timelike_candidates;	//Candidates for a timelike boundary
	int *chaintime;			//Maximal chain length to each node

	double *actth;			//Theoretical values of action for different densities

	Bitvector longest_chains;	//Longest chains' elements
	int longest_chain_length;	//Longest (maximum) chain
	std::pair<int,int> longest_pair;//Pair of elements bounding maximum chain

	float hub_density;		//Density of Hubs
	float *hub_densities;		//Density as a Function of Number of Hubs

	float geo_discon;		//Fraction of geodesically disconnected pairs

	Bitvector timelike_foliation;	//Set of chains
	Bitvector spacelike_foliation;	//Set of antichains
	std::vector<unsigned int> ax_set;	//Size of Alexandroff set of extremal elements

	float dimension;		//Estimate of dimension (Myrheim-Meyer)

	uint64_t *smi;			//Spacetime Mutual Information Cardinalities;

	int *layers;			//Proper time, as defined by layering
	float extrinsic_curvature;	//Mean extrinsic curvature of timelike boundaries
	std::vector<std::tuple<FastBitset, int, int>> boundary_chains;
};

//Network object containing minimal unique information
struct Network {
	Network() : network_properties(NetworkProperties()), network_observables(NetworkObservables()), nodes(Node()), edges(Edge()) {}
	Network(NetworkProperties _network_properties) : network_properties(_network_properties), network_observables(NetworkObservables()), nodes(Node()), edges(Edge()) {}
	Network(const Network &_network) : network_properties(_network.network_properties), network_observables(_network.network_observables), nodes(Node()), edges(Edge()) {}

	NetworkProperties network_properties;
	NetworkObservables network_observables;

	Node nodes;
	Edge edges;
	Bitvector adj;
	Bitvector links;

	std::vector<unsigned> U;
	std::vector<unsigned> V;
};

//Algorithmic Performance
struct CausetPerformance {
	CausetPerformance() : sCauset(Stopwatch()), sCalcDegrees(Stopwatch()), sCreateNetwork(Stopwatch()), sGenerateNodes(Stopwatch()), sGenerateNodesGPU(Stopwatch()), sGenKR(Stopwatch()), sGenRandom(Stopwatch()), sQuicksort(Stopwatch()), sLinkNodes(Stopwatch()), sLinkNodesGPU(Stopwatch()), sMeasureClustering(Stopwatch()), sMeasureConnectedComponents(Stopwatch()), sMeasureSuccessRatio(Stopwatch()), sMeasureAction(Stopwatch()), sMeasureActionTimelike(Stopwatch()), sMeasureThAction(Stopwatch()), sMeasureChain(Stopwatch()), sMeasureHubs(Stopwatch()), sMeasureGeoDis(Stopwatch()), sMeasureFoliation(Stopwatch()), sMeasureAntichain(Stopwatch()), sMeasureDimension(Stopwatch()), sMeasureEntanglementEntropy(Stopwatch()), sMeasureSMI(Stopwatch()), sMeasureExtrinsicCurvature(Stopwatch()) {}

	Stopwatch sCauset;
	Stopwatch sCalcDegrees;
	Stopwatch sCreateNetwork;
	Stopwatch sGenerateNodes;
	Stopwatch sGenerateNodesGPU;
	Stopwatch sGenKR;
	Stopwatch sGenRandom;
	Stopwatch sQuicksort;
	Stopwatch sLinkNodes;
	Stopwatch sLinkNodesGPU;
	Stopwatch sMeasureClustering;
	Stopwatch sMeasureConnectedComponents;
	Stopwatch sMeasureSuccessRatio;
	Stopwatch sMeasureAction;
	Stopwatch sMeasureActionTimelike;
	Stopwatch sMeasureThAction;
	Stopwatch sMeasureChain;
	Stopwatch sMeasureHubs;
	Stopwatch sMeasureGeoDis;
	Stopwatch sMeasureFoliation;
	Stopwatch sMeasureAntichain;
	Stopwatch sMeasureDimension;
	Stopwatch sMeasureEntanglementEntropy;
	Stopwatch sMeasureSMI;
	Stopwatch sMeasureExtrinsicCurvature;
};

//Benchmark Statistics
struct Benchmark {
	Benchmark() : bCalcDegrees(0.0), bCreateNetwork(0.0), bGenerateNodes(0.0), bGenerateNodesGPU(0.0), bGenKR(0.0), bGenRandom(0.0), bQuicksort(0.0), bLinkNodes(0.0), bLinkNodesGPU(0.0), bMeasureClustering(0.0), bMeasureConnectedComponents(0.0), bMeasureSuccessRatio(0.0), bMeasureAction(0.0), bMeasureChain(0.0), bMeasureHubs(0.0), bMeasureGeoDis(0.0), bMeasureFoliation(0.0), bMeasureAntichain(0.0), bMeasureDimension(0.0), bMeasureEntanglementEntropy(0.0), bMeasureSMI(0.0), bMeasureExtrinsicCurvature(0.0) {}

	double bCalcDegrees;
	double bCreateNetwork;
	double bGenerateNodes;
	double bGenerateNodesGPU;
	double bGenKR;
	double bGenRandom;
	double bQuicksort;
	double bLinkNodes;
	double bLinkNodesGPU;
	double bMeasureClustering;
	double bMeasureConnectedComponents;
	double bMeasureSuccessRatio;
	double bMeasureAction;
	double bMeasureChain;
	double bMeasureHubs;
	double bMeasureGeoDis;
	double bMeasureFoliation;
	double bMeasureAntichain;
	double bMeasureDimension;
	double bMeasureEntanglementEntropy;
	double bMeasureSMI;
	double bMeasureExtrinsicCurvature;
};

//Custom exception class used in this program
class CausetException : public std::exception
{
public:
	CausetException() : msg("Unknown Error!") {}
	explicit CausetException(char const * _msg) : msg(_msg) {}
	virtual ~CausetException() throw () {}
	virtual const char * what() const throw () { return msg; }

protected:
	char const * msg;
};

//Function prototypes for those described in src/Causet.cu
NetworkProperties parseArgs(int argc, char **argv, CausetMPI *cmpi);

void printModules(const int mpi_procs, const int rank);

bool initializeNetwork(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, CUcontext &ctx, const size_t dev_memory);

bool annealNetwork(Network * const network, CuResources * const cu, CaResources * const ca);

bool measureNetworkObservables(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, const CUcontext &ctx);

bool loadNetwork(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, CUcontext &ctx, const size_t dev_memory);

bool printNetworkHDF5(Network &network, CausetPerformance &cp);

bool printNetwork(Network &network, CausetPerformance &cp);

bool printBenchmark(const Benchmark &bm, const CausetFlags &cf, const bool &link, const bool &relink);

void destroyNetwork(Network * const network, size_t &hostMemUsed, size_t &devMemUsed);

bool linkNodes(Network * const network, CaResources * const ca, CausetPerformance * const cp, CUcontext &ctx);

#endif
