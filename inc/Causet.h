#ifndef CAUSET_H_
#define CAUSET_H_

#ifdef AVX2_ENABLED
#include <intrinsics/x86intrin.h>
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
#include <pthread.h>
#include <sstream>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unordered_set>
#include <vector>

//System Files for Parallel Acceleration
#ifdef CUDA_ENABLED
  #include <cuda.h>
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
#include <gsl/gsl_sf_lambert.h>
#include <sys/io.h>

//Custom System Files
#include <fastmath/inc/FastBitset.h>
#include <fastmath/inc/FastMath.h>
#include <fastmath/inc/FastNumInt.h>
#include <fastmath/inc/stopwatch.h>
#include <printcolor/printcolor.h>

//Local Files
#include "autocorr2.h"
#include "Constants.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

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

//Bitsets
typedef std::vector< FastBitset > Bitvector;

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
typedef boost::variate_generator< Engine &, UDistribution > UGenerator;
typedef boost::variate_generator< Engine &, NDistribution > NGenerator;
typedef boost::variate_generator< Engine &, PDistribution > PGenerator;

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

//A brief explanation of the Manifold, Region, Curvature,
//and Symmetry enums: These values are chosen and chained
//together so in the code you can create a unique integer
//which defines a spacetime by writing
// > int spacetime = dim | manifold | region | curvature | symmetry

//NOTE: This assumes dim = {2, 3, 4} only
//Make sure if you add to these you make sure the 'spacetime'
//value still fits inside a 32-bit integer.  Otherwise
//use a uint64_t variable.

//NOTE: If you modify any of these enums, increment the
//`VERSION' variable located in Constants.h

//Manifold Types
enum Manifold {
	ManifoldFirst	= 1 << 3,
	MINKOWSKI	= ManifoldFirst,
	MILNE		= MINKOWSKI << 1,
	DE_SITTER	= MILNE << 1,
	DUST		= DE_SITTER << 1,
	FLRW		= DUST << 1,
	HYPERBOLIC	= FLRW << 1,
	ManifoldLast	= HYPERBOLIC
};

//Region Types
enum Region {
	RegionFirst	= ManifoldLast << 1,
	SLAB		= RegionFirst,
	HALF_DIAMOND	= SLAB << 1,
	DIAMOND		= HALF_DIAMOND << 1,
	SAUCER		= DIAMOND << 1,
	RegionLast	= SAUCER
};

//Spatial Curvature
enum Curvature {
	CurvatureFirst	= RegionLast << 1,
	FLAT		= CurvatureFirst,
	POSITIVE	= FLAT << 1,
	CurvatureLast	= POSITIVE
};

//Temporal Symmetry
enum Symmetry {
	SymmetryFirst	= CurvatureLast << 1,
	ASYMMETRIC	= SymmetryFirst,
	SYMMETRIC	= ASYMMETRIC << 1,
	SymmetryLast	= SYMMETRIC
};

static const std::string manifoldNames[] = { "Minkowski", "Milne", "De Sitter", "Dust", "FLRW", "Hyperbolic" };
static const std::string regionNames[] = { "Slab", "Half-Diamond", "Diamond", "Saucer" };
static const std::string curvatureNames[] = { "Flat", "Positive" };
static const std::string symmetryNames[] = { "Asymmetric", "Symmetric" };

static const unsigned int stdim_mask = ManifoldFirst - 1;
static const unsigned int manifold_mask = (ManifoldLast << 1) - ManifoldFirst;
static const unsigned int region_mask = (RegionLast << 1) - RegionFirst;
static const unsigned int curvature_mask = (CurvatureLast << 1) - CurvatureFirst;
static const unsigned int symmetry_mask = (SymmetryLast << 1) - SymmetryFirst;

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

//Embedding Validation Data
struct EVData {
	EVData() : confusion(NULL), A1T(0.0), A1S(0.0) {}

	uint64_t *confusion;		//Confusion Matrix

	double A1T;			//Normalization for timelike distances
	double A1S;			//Normalization for spacelike distances
};

//Distance Validation Data
struct DVData {
	DVData() : confusion(NULL), norm(0.0) {}

	uint64_t *confusion;		//Confusion Matrix

	double norm;			//Normalization constant
};

enum CausetSpinlock {
	UNLOCKED,
	LOCKED
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
	CausetFlags() : use_gpu(false), decode_cpu(false), print_network(false), print_edges(false), link(false), relink(false), read_old_format(false), quiet_read(false), no_pos(false), use_bit(false), gen_ds_table(false), gen_flrw_table(false), calc_clustering(false), calc_components(false), calc_success_ratio(false), calc_autocorr(false), calc_deg_field(false), calc_action(false), validate_embedding(false), validate_distances(false), verbose(false), bench(false), yes(false), test(false) {}

	bool use_gpu;			//Use GPU to Accelerate Select Algorithms
	bool decode_cpu;		//Decode edge list using serial sort
	bool print_network;		//Print to File
	bool print_edges;		//Print Edge List to File
	bool link;			//Link Nodes after Generation
	bool relink;			//Link Nodes in Graph Identified by 'graphID'

	bool read_old_format;		//Read Node Positions in the Format (theta3, theta2, theta1)
	bool quiet_read;		//Ignore Warnings when Reading Graph
	bool no_pos;			//No positions in graph (edge list only)
	bool use_bit;			//Use bit array instead of sparse edge lists
	bool gen_ds_table;		//Generate de Sitter geodesic lookup table
	bool gen_flrw_table;		//Generate FLRW geodesic lookup table
	
	bool calc_clustering;		//Find Clustering Coefficients
	bool calc_components;		//Find Connected Components
	bool calc_success_ratio;	//Find Success Ratio
	bool calc_autocorr;		//Autocorrelation
	bool calc_deg_field;		//Measure Degree Field
	bool calc_action;		//Measure Action

	bool validate_embedding;	//Find Embedding Statistics
	bool validate_distances;	//Compare Distance Methods
	
	bool verbose;			//Verbose Output
	bool bench;			//Benchmark Algorithms
	bool yes;			//Suppresses User Input
	bool test;			//Test Parameters
};

//Numerical parameters constraining the network
struct NetworkProperties {
	NetworkProperties() : flags(CausetFlags()), spacetime(0), N_tar(0), k_tar(0.0), N_emb(0.0), N_sr(0.0), N_df(0), tau_m(0.0), N_dst(0.0), max_cardinality(0), a(0.0), eta0(0.0), zeta(0.0), zeta1(0.0), r_max(0.0), tau0(0.0), alpha(0.0), delta(0.0), omegaM(0.0), omegaL(0.0), core_edge_fraction(0.01), edge_buffer(0.0), seed(12345L), graphID(0), cmpi(CausetMPI()), mrng(MersenneRNG()), group_size(1) {}

	CausetFlags flags;
	unsigned int spacetime;		//Spacetime Definition
					//Encodes dimension, manifold, region, curvature, and symmetry

	int N_tar;			//Target Number of Nodes
	float k_tar;			//Target Average Degree

	long double N_emb;		//Number of Pairs Used in Embedding Validation
	long double N_sr;		//Number of Pairs Used in Success Ratio
	int N_df;			//Number of Samples Used in Degree Field Measurements
	double tau_m;			//Rescaled Time of Nodes used for Measuring Degree Field
	double N_dst;			//Number of Pairs Used in Distance Validation
	int max_cardinality;		//Elements used in Action Calculation

	double a;			//Hyperboloid Pseudoradius
	double eta0;			//Maximum Conformal Time
	double zeta;			//Pi/2 - Eta_0
	double zeta1;			//Pi/2 - Eta_1

	double r_max;			//Size of the Spatial Slice (Radius)
	double tau0;			//Rescaled Age
	double alpha;			//Rescaled Ratio of Matter Density to Dark Energy Density
	double delta;			//Node Density

	double omegaM;			//Matter Density
	double omegaL;			//Dark Energy Density

	float core_edge_fraction;	//Fraction of nodes designated as having core edges
	float edge_buffer;		//Fraction of edge list added as a buffer

	long seed;			//Random Seed
	int graphID;			//Unique Simulation ID

	CausetMPI cmpi;			//MPI Flags

	MersenneRNG mrng;		//Mersenne Twister RNG

	int group_size;			//Number of mega-blocks per grid dimension
};

//Measured values of the network
struct NetworkObservables {
	NetworkObservables() : N_res(0), k_res(0.0f), N_deg2(0), N_cc(0), N_gcc(0), clustering(NULL), average_clustering(0.0), evd(EVData()), success_ratio(0.0), success_ratio2(0.0), in_degree_field(NULL), avg_idf(0.0), out_degree_field(NULL), avg_odf(0.0), dvd(DVData()), cardinalities(NULL), action(0.0f) {}
	
	int N_res;			//Resulting Number of Connected Nodes
	float k_res;			//Resulting Average Degree

	int N_deg2;			//Nodes of Degree 2 or Greater

	int N_cc;			//Number of Connected Components
	int N_gcc;			//Size of Giant Connected Component

	float *clustering;		//Clustering Coefficients
	float average_clustering;	//Average Clustering over All Nodes

	EVData evd;			//Embedding Validation Data

	float success_ratio;		//Success Ratio (type 1 - all pairs which are geodesically connected are considered)
	float success_ratio2;		//Success Ratio (type 2 - pairs which are geodesically disconnected, but have neighbors which are connected, are considered here)

	int *in_degree_field;		//In-Degree Field Measurements
	float avg_idf;			//Average In-Degree Field Value

	int *out_degree_field;		//Out-Degree Field Measurements
	float avg_odf;			//Average Out-Degree Field Value

	DVData dvd;			//Distance Validation Data

	uint64_t *cardinalities;	//M-Element Inclusive-Order-Interval Cardinalities
	float action;			//Action
};

//Network object containing minimal unique information
struct Network {
	Network() : network_properties(NetworkProperties()), network_observables(NetworkObservables()), nodes(Node()), edges(Edge()) {}
	Network(NetworkProperties _network_properties) : network_properties(_network_properties), network_observables(NetworkObservables()), nodes(Node()), edges(Edge()) {}

	NetworkProperties network_properties;
	NetworkObservables network_observables;

	Node nodes;
	Edge edges;
	Bitvector adj;
};

//Algorithmic Performance
struct CausetPerformance {
	CausetPerformance() : sCauset(Stopwatch()), sCalcDegrees(Stopwatch()), sCreateNetwork(Stopwatch()), sGenerateNodes(Stopwatch()), sGenerateNodesGPU(Stopwatch()), sQuicksort(Stopwatch()), sLinkNodes(Stopwatch()), sLinkNodesGPU(Stopwatch()), sMeasureClustering(Stopwatch()), sMeasureConnectedComponents(Stopwatch()), sValidateEmbedding(Stopwatch()), sMeasureSuccessRatio(Stopwatch()), sMeasureDegreeField(Stopwatch()), sValidateDistances(Stopwatch()), sMeasureAction(Stopwatch()) {}

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
	Stopwatch sValidateEmbedding;
	Stopwatch sMeasureSuccessRatio;
	Stopwatch sMeasureDegreeField;
	Stopwatch sValidateDistances;
	Stopwatch sMeasureAction;
};

//Benchmark Statistics
struct Benchmark {
	Benchmark() : bCalcDegrees(0.0), bCreateNetwork(0.0), bGenerateNodes(0.0), bGenerateNodesGPU(0.0), bQuicksort(0.0), bLinkNodes(0.0), bLinkNodesGPU(0.0), bMeasureClustering(0.0), bMeasureConnectedComponents(0.0), bMeasureSuccessRatio(0.0), bMeasureDegreeField(0.0), bMeasureAction(0.0) {}

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
	double bMeasureDegreeField;
	double bMeasureAction;
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

bool initializeNetwork(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, const CUcontext &ctx);

bool measureNetworkObservables(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm);

bool loadNetwork(Network * const network, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm, const CUcontext &ctx);

bool printNetwork(Network &network, CausetPerformance &cp, const int &gpuID);

bool printBenchmark(const Benchmark &bm, const CausetFlags &cf, const bool &link, const bool &relink);

void destroyNetwork(Network * const network, size_t &hostMemUsed, size_t &devMemUsed);

//Useful utility functions
inline unsigned int get_stdim(const unsigned int &spacetime)  { return (unsigned int)(spacetime & stdim_mask);  }
inline Manifold get_manifold(const unsigned int &spacetime)   { return (Manifold)(spacetime & manifold_mask);   }
inline Region get_region(const unsigned int &spacetime)       { return (Region)(spacetime & region_mask);	}
inline Curvature get_curvature(const unsigned int &spacetime) { return (Curvature)(spacetime & curvature_mask); }
inline Symmetry get_symmetry(const unsigned int &spacetime)   { return (Symmetry)(spacetime & symmetry_mask);   }

#endif
