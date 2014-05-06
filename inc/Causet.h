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

#include <curand.h>
#include <GL/freeglut.h>

#include "autocorr2.h"
#include "CuResources.h"
#include "ran2.h"

#define TOL (1e-28)	//Any value smaller than this is rounded to zero
#define NPRINT 10000	//Used for debugging statements in loops
#define NBENCH 10	//Times each function is run during benchmarking

bool CAUSET_DEBUG = false;	//Activates certain print statements for verbose output
bool BENCH = false;		//Activates benchmarking of selected routines

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
	float t;

	//Spatial Coordinates
	float theta;
	float phi;
	float chi;

	//Neighbors
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
	CausetFlags() : cc(CausetConflicts()), use_gpu(false), disp_network(false), print_network(false), universe(false), calc_clustering(false), calc_autocorr(false) {}
	CausetFlags(CausetConflicts _cc, bool _use_gpu, bool _disp_network, bool _print_network, bool _universe, bool _calc_clustering, bool _calc_autocorr) : cc(_cc), use_gpu(_use_gpu), disp_network(_disp_network), print_network(_print_network), universe(_universe), calc_clustering(_calc_clustering), calc_autocorr(_calc_autocorr) {}

	CausetConflicts cc;	//Conflicting Parameters

	bool use_gpu;		//Use GPU to Accelerate Select Algorithms
	bool disp_network;	//Plot Network using OpenGL
	bool print_network;	//Print to File
	bool universe;		//Use Universe's Tau Distribution

	bool calc_clustering;	//Find Clustering Coefficients
	bool calc_autocorr;	//Autocorrelation
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
	Manifold manifold;		//Manifold of the Network

	int N_tar;			//Target Number of Nodes
	float k_tar;			//Target Average Degree

	int N_res;			//Resulting Number of Connected Nodes
	float k_res;			//Resulting Average Degree

	int N_deg2;			//Nodes of Degree 2 or Greater

	int dim;			//Spacetime Dimension (2 or 4)

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

//Function prototypes for those described in src/Causet.cu
bool initializeNetwork(Network *network, CausetPerformance *cp, Benchmark *bm);
void measureNetworkObservables(Network *network, CausetPerformance *cp, Benchmark *bm);
bool displayNetwork(Node *nodes, int *future_edges, int argc, char **argv);
void display();
bool loadNetwork(Network *network, CausetPerformance *cp);
bool printNetwork(Network network, CausetPerformance cp, long init_seed);
bool printBenchmark(Benchmark bm, CausetFlags cf);
void destroyNetwork(Network *network);

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

//Parse Command Line Arguments
NetworkProperties parseArgs(int argc, char **argv)
{
	NetworkProperties network_properties = NetworkProperties();

	//Initialize conflict array to zeros (no conflicts)
	for (int i = 0; i < 7; i++)
		network_properties.flags.cc.conflicts[i] = 0;

	int c, longIndex;
	//Single-character options
	static const char *optString = ":m:n:k:d:s:a:c:g:t:A:D:o:r:l:uvCh";
	//Multi-character options
	static const struct option longOpts[] = {
		{ "manifold",	required_argument,	NULL, 'm' },
		{ "nodes", 	required_argument,	NULL, 'n' },
		{ "degrees",	required_argument,	NULL, 'k' },
		{ "dim",	required_argument,	NULL, 'd' },
		{ "seed",	required_argument,	NULL, 's' },
		{ "clustering",	no_argument,		NULL, 'C' },
		{ "graph",	required_argument,	NULL, 'g' },
		{ "universe",	no_argument,		NULL, 'u' },
		{ "age",	required_argument,	NULL, 't' },
		{ "alpha",	required_argument,	NULL, 'A' },
		{ "delta",	required_argument,	NULL, 'D' },
		{ "energy",	required_argument,	NULL, 'o' },
		{ "ratio",	required_argument,	NULL, 'r' },
		{ "lambda",	required_argument,	NULL, 'l' },

		{ "help", 	no_argument,		NULL, 'h' },
		{ "gpu", 	no_argument, 		NULL,  0  },
		{ "display", 	no_argument, 		NULL,  0  },
		{ "print", 	no_argument, 		NULL,  0  },
		{ "verbose", 	no_argument, 		NULL, 'v' },
		{ "benchmark",	no_argument,		NULL,  0  },
		{ "autocorr",	no_argument,		NULL,  0  },
		{ "confliicts", no_argument,		NULL,  0  },
		{ NULL,		0,			0,     0  }
	};

	try {
		while ((c = getopt_long(argc, argv, optString, longOpts, &longIndex)) != -1) {
			switch (c) {
			case 'm':	//Manifold
				if (strcmp(optarg, "e"))
					network_properties.manifold = EUCLIDEAN;
				else if (strcmp(optarg, "d"))
					network_properties.manifold = DE_SITTER;
				else if (strcmp(optarg, "a"))
					network_properties.manifold = ANTI_DE_SITTER;
				else
					throw CausetException("Invalid argument for 'manifold' parameter!\n");

				if (network_properties.manifold != DE_SITTER) {
					printf("Only de Sitter manifold currently supported!  Reverting to default value.\n");
					network_properties.manifold = DE_SITTER;
				}

				break;
			case 'n':	//Number of nodes
				network_properties.N_tar = atoi(optarg);
				if (network_properties.N_tar <= 0)
					throw CausetException("Invalid argument for 'nodes' parameter!\n");

				network_properties.flags.cc.conflicts[4]++;
				network_properties.flags.cc.conflicts[5]++;
				network_properties.flags.cc.conflicts[6]++;

				break;
			case 'k':	//Average expected degrees
				network_properties.k_tar = atof(optarg);
				if (network_properties.k_tar <= 0.0)
					throw CausetException("Invalid argument for 'degrees' parameter!\n");
				break;
			case 'd':	//Spatial dimensions
				network_properties.dim = atoi(optarg);
				if (!(atoi(optarg) == 1 || atoi(optarg) == 3))
					throw CausetException("Invalid argument for 'dimension' parameter!\n");
				break;
			case 's':	//Random seed
				network_properties.seed = -1.0 * atol(optarg);
				if (network_properties.seed >= 0.0L)
					throw CausetException("Invalid argument for 'seed' parameter!\n");
				break;
			case 'a':	//Pseudoradius
				network_properties.a = atof(optarg);
				
				if (network_properties.a <= 0.0)
					throw CausetException("Invalid argument for 'a' parameter!\n");

				network_properties.lambda = 3.0 / powf(network_properties.a, 2.0);

				network_properties.flags.cc.conflicts[0]++;

				break;
			case 'c':	//Core edge fraction (used for adjacency matrix)
				network_properties.core_edge_fraction = atof(optarg);
				if (network_properties.core_edge_fraction <= 0.0 || network_properties.core_edge_fraction >= 1.0)
					throw CausetException("Invalid argument for 'c' parameter!\n");
				break;
			case 'C':	//Flag for calculating clustering
				network_properties.flags.calc_clustering = true;
				break;
			case 'g':	//Graph ID
				network_properties.graphID = atoi(optarg);
				if (network_properties.graphID <= 0)
					throw CausetException("Invalid argument for 'Graph ID' parameter!\n");
				break;
			case 'u':	//Flag for creating universe causet
				network_properties.flags.universe = true;
				break;
			case 't':	//Age of universe
				network_properties.tau0 = atof(optarg);

				if (network_properties.tau0 <= 0.0)
					throw CausetException("Invalid argument for 'age' parameter!\n");

				network_properties.ratio = powf(sinh(1.5 * network_properties.tau0), 2.0);
				network_properties.omegaM = 1.0 / (network_properties.ratio + 1.0);
				network_properties.omegaL = 1.0 - network_properties.omegaM;

				network_properties.flags.cc.conflicts[2]++;
				network_properties.flags.cc.conflicts[3]++;
				network_properties.flags.cc.conflicts[6]++;

				break;
			case 'A':	//Rescaled ratio of dark energy density to matter density
				network_properties.alpha = atof(optarg);

				if (network_properties.alpha <= 0.0)
					throw CausetException("Invalid argument for 'alpha' parameter!\n");

				network_properties.flags.cc.conflicts[4]++;
				network_properties.flags.cc.conflicts[5]++;
				network_properties.flags.cc.conflicts[6]++;

				break;
			case 'D':	//Density of nodes
				network_properties.delta = atof(optarg);

				if (network_properties.delta <= 0.0)
					throw CausetException("Invalid argument for 'delta' parameter!\n");

				network_properties.flags.cc.conflicts[4]++;
				network_properties.flags.cc.conflicts[5]++;
				network_properties.flags.cc.conflicts[6]++;

				break;
			case 'o':	//Density of dark energy
				network_properties.omegaL = atof(optarg);

				if (network_properties.omegaL <= 0.0 || network_properties.omegaL >= 1.0)
					throw CausetException("Invalid input for 'energy' parameter!\n");

				network_properties.omegaM = 1.0 - network_properties.omegaL;
				network_properties.ratio = network_properties.omegaL / network_properties.omegaM;
				network_properties.tau0 = (2.0 / 3.0) * asinh(sqrt(network_properties.ratio));
					
				network_properties.flags.cc.conflicts[1]++;
				network_properties.flags.cc.conflicts[2]++;
				network_properties.flags.cc.conflicts[5]++;

				break;
			case 'r':	//Ratio of dark energy density to matter density
				network_properties.ratio = atof(optarg);

				if (network_properties.ratio <= 0.0)
					throw CausetException("Invalid argument for 'ratio' parameter!\n");

				network_properties.tau0 = (2.0 / 3.0) * asinh(sqrt(network_properties.ratio));
				network_properties.omegaM = 1.0 / (network_properties.ratio + 1.0);
				network_properties.omegaL = 1.0 - network_properties.omegaM;
				
				network_properties.flags.cc.conflicts[1]++;
				network_properties.flags.cc.conflicts[3]++;
				network_properties.flags.cc.conflicts[4]++;

				break;
			case 'l':	//Cosmological constant
				network_properties.lambda = atof(optarg);

				if (network_properties.lambda <= 0.0)
					throw CausetException("Invalid argument for 'lambda' parameter!\n");

				network_properties.a = sqrt(3.0 / network_properties.lambda);

				network_properties.flags.cc.conflicts[0]++;

				break;
			case 'v':	//Verbose output
				CAUSET_DEBUG = true;
				break;
			case 0:
				if (strcmp("gpu", longOpts[longIndex].name) == 0)
					//Flag to use GPU accelerated routines
					network_properties.flags.use_gpu = true;
				else if (strcmp("display", longOpts[longIndex].name) == 0)
					//Flag to use OpenGL to display network
					//network_properties.flags.disp_network = true;
					printf("Display not supported:  Ignoring Flag.\n");
				else if (strcmp("print", longOpts[longIndex].name) == 0)
					//Flag to print results to file in 'dat' folder
					network_properties.flags.print_network = true;
				else if (strcmp("benchmark", longOpts[longIndex].name) == 0)
					//Flag to benchmark selected routines
					BENCH = true;
				else if (strcmp("autocorr", longOpts[longIndex].name) == 0)
					//Flag to calculate autocorrelation of selected variables
					network_properties.flags.calc_autocorr = true;
				else if (strcmp("conflicts", longOpts[longIndex].name) == 0) {
					//Print conflicting parameters
					printf("\nParameter Conflicts:\n");
					printf("--------------------\n");
					printf(" > a, lambda\n");
					printf(" > energy, ratio\n");
					printf(" > energy, age\n");
					printf(" > age, ratio\n");
					printf(" > n, delta, alpha, ratio\n");
					printf(" > n, delta, alpha, energy\n");
					printf(" > n, delta, alpha, age\n\n");
					printf("Specifying any of these combinations will over-constrain the system!\n\n");
					exit(EXIT_SUCCESS);
				} else {
					//Unrecognized options
					fprintf(stderr, "Option --%s is not recognized.\n", longOpts[longIndex].name);
					exit(EXIT_FAILURE);
				}
				break;
			case 'h':
				//Print help menu
				printf("\nUsage  :  CausalSet [options]\n\n");
				printf("CausalSet Options...................\n");
				printf("====================================\n");
				printf("Flag:\t\t\tVariable:\t\t\tSuggested Values:\n");
				printf("  -A, --alpha\t\tUnphysical Parameter\t\t2.0\n");
				printf("  -a\t\t\tPseudoradius\t\t\t1.0\n");
				printf("  -C, --clustering\tCalculate Clustering\n");
				printf("  -c\t\t\tCore Edge Ratio\t\t\t0.01\n");
				printf("  -D, --delta\t\tNode Density\t\t\t10000\n");
				printf("  -d, --dim\t\tSpatial Dimensions\t\t1 or 3\n");
				printf("  -g, --graph\t\tGraph ID\t\t\tCheck dat/*.cset.out files\n");
				printf("  -h, --help\t\tDisplay this menu\n");
				printf("  -k, --degrees\t\tExpected Average Degrees\t10-100\n");
				printf("  -l, --lambda\t\tCosmological Constant\t\t3.0\n");
				printf("  -m, --manifold\tManifold\t\t\tEUCLIDEAN, DE_SITTER, ANTI_DE_SITTER\n");
				printf("  -n, --nodes\t\tNumber of Nodes\t\t\t100-100000\n");
				printf("  -o, --energy\t\tDark Energy Density\t\t0.73\n");
				printf("  -r, --ratio\t\tEnergy to Matter Ratio\t\t2.7\n");
				printf("  -s, --seed\t\tRNG Seed\t\t\t18100L\n");
				printf("  -t, --age\t\tRescaled Age of Universe\t0.85\n");
				printf("  -u, --universe\tUniverse Causet\n");
				printf("  -v, --verbose\t\tVerbose Output\n");
				printf("\n");

				printf("Flag:\t\t\tPurpose:\n");
				printf("  --autocorr\t\tCalculate Autocorrelations\n");
				printf("  --benchmark\t\tBenchmark Algorithms\n");
				printf("  --conflicts\t\tShow Parameter Conflicts\n");
				printf("  --display\t\tDisplay Graph\n");
				printf("  --gpu\t\t\tUse GPU\n");
				printf("  --print\t\tPrint Results\n");
				printf("\n");
				exit(EXIT_SUCCESS);
			case ':':
				//Single-character flag needs an argument
				fprintf(stderr, "%s: option '-%c' requires an argument\n", argv[0], optopt);
				exit(EXIT_FAILURE);
			case '?':	//Unrecognized flag
			default:	//Default case
				fprintf(stderr, "%s:option -%c' is not recognized.\n", argv[0], optopt);
				exit(EXIT_FAILURE);
			}
		}

		//Make sure necessary parameters have been specified
		if (!network_properties.flags.universe) {
			if (network_properties.N_tar == 0)
				throw CausetException("Flag '-n', number of nodes, must be specified!\n");
			else if (network_properties.k_tar == 0.0)
				throw CausetException("Flag '-k', expected average degrees, must be specified!\n");
		}

		//Prepare to benchmark algorithms
		if (BENCH) {
			CAUSET_DEBUG = false;
			network_properties.graphID = 0;
			network_properties.flags.disp_network = false;
			network_properties.flags.print_network = false;
		}

		//If no seed specified, choose random one
		if (network_properties.seed == -12345L) {
			srand(time(NULL));
			network_properties.seed = -1.0 * (long)time(NULL);
		}

		//If graph ID specified, prepare to read graph properties
		if (network_properties.graphID != 0) {
			if (CAUSET_DEBUG) {
				printf("You have chosen to load a graph from memory.  Some parameters may be ignored as a result.  Continue [y/N]? ", network_properties.graphID);
				char response = getchar();
				if (response != 'y')
				exit(EXIT_FAILURE);
			}

			//Not currently supported for 1+1 causet
			network_properties.dim = 3;
		}
	} catch (CausetException c) {
		fprintf(stderr, "CausetException in %s: %s on line %d\n", __FILE__, c.what(), __LINE__);
		exit(EXIT_FAILURE);
	} catch (std::exception e) {
		fprintf(stderr, "Unknown Exception in %s: %s on line %d\n", __FILE__, e.what(), __LINE__);
		exit(EXIT_FAILURE);
	}

	return network_properties;
}

#endif
