#ifndef CAUSET_H_
#define CAUSET_H_

#include <exception>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>

#include <curand.h>
#include <GL/freeglut.h>

#include "CuResources.h"
#include "ran2.h"

#define TOL (1e-28)
#define NPRINT 10000

bool CAUSET_DEBUG = false;

////////////////////////////////////////////////////////////////////////////
//References
//[1] Network Cosmology
//    http://www.nature.com/srep/2012/121113/srep00793/full/srep00793.html
//[2] Supplementary Information for Network Cosmology
//    http://complex.ffn.ub.es/~mbogunya/archivos_cms/files/srep00793-s1.pdf
////////////////////////////////////////////////////////////////////////////

struct Node {
	Node() : tau(0.0), theta(0.0), phi(0.0), chi(0.0), num_in(0), num_out(0) {}
	Node(float _tau, float _theta, float _phi, float _chi, unsigned int _num_in, unsigned int _num_out) : tau(_tau), theta(_theta), phi(_phi), chi(_chi), num_in(_num_in), num_out(_num_out) {}

	//Temporal Coordinate
	float tau;

	//Spatial Coordinates
	float theta;
	float phi;
	float chi;

	//Neighbors
	unsigned int num_in;
	unsigned int num_out;
};

struct CausetFlags {
	CausetFlags() : use_gpu(false), disp_network(false), print_network(false), calc_clustering(false) {}
	CausetFlags(bool _use_gpu, bool _disp_network, bool _print_network, bool _calc_clustering) : use_gpu(_use_gpu), disp_network(_disp_network), print_network(_print_network), calc_clustering(_calc_clustering) {}

	bool use_gpu;			//Use GPU to Accelerate Select Algorithms
	bool disp_network;		//Plot Network using OpenGL
	bool print_network;		//Print to File

	bool calc_clustering;	//Find Clustering Coefficients
};

//CUDA Kernel Execution Parameters
struct NetworkExec {
	NetworkExec() : threads_per_block(dim3(256, 256, 1)), blocks_per_grid(dim3(256, 256, 1)) {}
	NetworkExec(dim3 tpb, dim3 bpg) : threads_per_block(tpb), blocks_per_grid(bpg) {}

	dim3 threads_per_block;
	dim3 blocks_per_grid;
};

struct NetworkProperties {
	NetworkProperties() : N_tar(10000), k_tar(10.0), N_res(0), k_res(0.0), N_deg2(0), dim(4), a(1.0), zeta(0.0), subnet_size(104857600), core_edge_ratio(0.01), edge_buffer(25000), seed(-12345L), flags(CausetFlags()), network_exec(NetworkExec()) {}
	NetworkProperties(unsigned int _N_tar, float _k_tar, unsigned int _dim, float _a, double _zeta, size_t _subnet_size, float _core_edge_ratio, unsigned int _edge_buffer, long _seed, CausetFlags _flags, NetworkExec _network_exec) : N_tar(_N_tar), k_tar(_k_tar), N_res(0), k_res(0), N_deg2(0), dim(_dim), a(_a), zeta(_zeta), subnet_size(_subnet_size), core_edge_ratio(_core_edge_ratio), edge_buffer(_edge_buffer), seed(_seed), flags(_flags), network_exec(_network_exec) {}

	CausetFlags flags;
	NetworkExec network_exec;

	unsigned int N_tar;		//Target Number of Nodes
	float k_tar;			//Target Average Degree

	unsigned int N_res;		//Resulting Number of Connected Nodes
	float k_res;			//Resulting Average Degree

	unsigned int N_deg2;		//Nodes of Degree 2 or Greater

	unsigned int dim;		//Spacetime Dimension (2 or 4)

	float a;			//Hyperboloid Pseudoradius
	double zeta;			//Pi/2 - Eta_0

	size_t subnet_size;		//Maximum Contiguous Memory Request (Default 100 MB)
	float core_edge_ratio;		//Fraction of nodes designated as having core edges
	unsigned int edge_buffer;	//Small memory buffer for adjacency list

	long seed;
};

struct NetworkObservables {
	NetworkObservables() : clustering(NULL), average_clustering(0.0) {}
	NetworkObservables(float *_clustering, float _average_clustering) : clustering(_clustering), average_clustering(_average_clustering) {}

	float *clustering;		//Clustering Coefficients
	float average_clustering;	//Average Clustering over All Nodes
};

struct Network {
	Network() : network_properties(NetworkProperties()), network_observables(NetworkObservables()), nodes(NULL), past_edges(NULL), future_edges(NULL), past_edge_row_start(NULL), future_edge_row_start(NULL), core_edge_exists(NULL) {}
	Network(NetworkProperties _network_properties) : network_properties(_network_properties), network_observables(NetworkObservables()), nodes(NULL), past_edges(NULL), future_edges(NULL), past_edge_row_start(NULL), future_edge_row_start(NULL), core_edge_exists(NULL) {}
	Network(NetworkProperties _network_properties, NetworkObservables _network_observables, Node *_nodes, unsigned int *_past_edges, unsigned int *_future_edges, int *_past_edge_row_start, int *_future_edge_row_start, bool *_core_edge_exists) : network_properties(_network_properties), network_observables(_network_observables), nodes(_nodes), past_edges(_past_edges), future_edges(_future_edges), past_edge_row_start(_past_edge_row_start), future_edge_row_start(_future_edge_row_start), core_edge_exists(_core_edge_exists) {}

	NetworkProperties network_properties;
	NetworkObservables network_observables;

	Node *nodes;
	unsigned int *past_edges;	//Sparse adjacency lists
	unsigned int *future_edges;
	int *past_edge_row_start;	//Adjacency list indices
	int *future_edge_row_start;
	bool *core_edge_exists;	//Adjacency matrix

	//GPU Memory Pointers
	CUdeviceptr d_nodes;
	CUdeviceptr d_edges;
};

bool initializeNetwork(Network *network);
void measureNetworkObservables(Network *network);
bool displayNetwork(Node *nodes, unsigned int *future_edges, int argc, char **argv);
void display();
bool printNetwork(Network network, long init_seed);
bool destroyNetwork(Network *network);

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

	int c, longIndex;
	static const char *optString = ":n:k:d:s:a:c:Ch";
	static const struct option longOpts[] = {
		{ "nodes", 	required_argument,	NULL, 'n' },
		{ "degrees",	required_argument,	NULL, 'k' },
		{ "dim",	required_argument,	NULL, 'd' },
		{ "seed",	required_argument,	NULL, 's' },
		{ "clustering",no_argument,		NULL, 'C' },

		{ "help", 	no_argument,		NULL, 'h' },
		{ "gpu", 	no_argument, 		NULL,  0  },
		{ "display", 	no_argument, 		NULL,  0  },
		{ "print", 	no_argument, 		NULL,  0  },
		{ "size", 	required_argument,	NULL,  0  },
		{ "debug", 	no_argument, 		NULL,  0  }
	};

	while ((c = getopt_long(argc, argv, optString, longOpts, &longIndex)) != -1) {
		switch (c) {
		case 'n':
			network_properties.N_tar = atoi(optarg);
			break;
		case 'k':
			network_properties.k_tar = atof(optarg);
			break;
		case 'd':
			try {
				if (!(atoi(optarg) == 2 || atoi(optarg) == 4))
					throw CausetException("Dimension not supported.\n");
				else
					network_properties.dim = atoi(optarg);
			} catch (CausetException e) {
				fprintf(stderr, e.what());
				exit(EXIT_FAILURE);
			}
			
			break;
		case 's':
			network_properties.seed = -1.0 * atol(optarg);
			break;
		case 'a':
			network_properties.a = atof(optarg);
			break;
		case 'c':
			network_properties.core_edge_ratio = atof(optarg);
			break;
		case 'C':
			network_properties.flags.calc_clustering = true;
			break;
		case 0:
			if (strcmp("gpu", longOpts[longIndex].name) == 0)
				network_properties.flags.use_gpu = true;
			else if (strcmp("display", longOpts[longIndex].name) == 0)
				//network_properties.flags.disp_network = true;
				printf("Display not supported:  Ignoring Flag.\n");
			else if (strcmp("print", longOpts[longIndex].name) == 0)
				network_properties.flags.print_network = true;
			else if (strcmp("size", longOpts[longIndex].name) == 0)
				network_properties.subnet_size = size_t(atof(optarg) * 1048576);
			else if (strcmp("debug", longOpts[longIndex].name) == 0)
				CAUSET_DEBUG = true;
			break;
		case 'h':
			printf("\nUsage  : CausalSet [options]\n\n");
			printf("CausalSet Options...................\n");
			printf("====================================\n");
			printf("Flag:\t\t\tVariable:\t\t\tSuggested Values:\n");
			printf("  -n, --nodes\t\tNumber of Nodes\t\t\t100-100000\n");
			printf("  -k, --degrees\t\tExpected Average Degrees\t10-100\n");
			printf("  -d, --dim\t\tSpacetime Dimensions\t\t2 or 4\n");
			printf("  -s, --seed\t\tRNG Seed\t\t\t12345L\n");
			printf("  -a\t\t\tPseudoradius\t\t\t~1.0\n");
			printf("  -c\t\t\tCore Edge Ratio\t\t\t~0.01\n");
			printf("  -C, --clustering\tCalculate Clustering\n");
			printf("  -h, --help\t\tDisplay this menu\n");

			printf("\n");
			printf("Flag:\t\t\tPurpose:\n");
			printf("  --gpu\t\t\tUse GPU\n");
			printf("  --print\t\tPrint Results\n");
			printf("  --display\t\tDisplay Graph\n");
			printf("  --size\t\tNot Implemented Yet!\n");
			printf("  --debug\t\tActivate Debug Statements\n");
			printf("\n");
			exit(EXIT_SUCCESS);
		case ':':
			fprintf(stderr, "%s: option '-%c' requires an argument\n", argv[0], optopt);
			exit(EXIT_FAILURE);
		case '?':
		default:
			fprintf(stderr, "%s:option -%c' is not recognized.\n", argv[0], optopt);
			exit(EXIT_FAILURE);
		}
	}

	if (!CAUSET_DEBUG) {
		srand(time(NULL));
		network_properties.seed = -1.0 * (long)time(NULL);
	}

	if (!network_properties.flags.calc_clustering && !CAUSET_DEBUG) {
		printf("You have not chosen to measure any observables!  Continue [y/N]? ");
		char response = getchar();
		if (response != 'y')
			exit(EXIT_FAILURE);
	} 

	return network_properties;
}

#endif
