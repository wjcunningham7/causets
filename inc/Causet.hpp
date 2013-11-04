#ifndef CAUSET_HPP_
#define CAUSET_HPP_

#include <exception>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>

#include "CuResources.hpp"
#include "ran2.h"

#define TOL (10^-6)

struct Node {
	Node() : eta(0.0), theta(0.0), phi(0.0), chi(0.0), num_in(0), num_out(0) {}
	Node(float _eta, float _theta, float _phi, float _chi, unsigned int _num_in, unsigned int _num_out) : eta(_eta), theta(_theta), phi(_phi), chi(_chi), num_in(_num_in), num_out(_num_out) {}

	//Temporal Coordinate
	float eta;

	//Spatial Coordinates
	float theta;
	float phi;
	float chi;

	//Neighbors
	unsigned int num_in;
	unsigned int num_out;
};

struct CausetFlags {
	CausetFlags() : use_gpu(false), disp_network(false), print_network(false) {}
	CausetFlags(bool _use_gpu, bool _disp_network, bool _print_network) : use_gpu(_use_gpu), disp_network(_disp_network), print_network(_print_network) {}

	bool use_gpu;		//Use GPU to Accelerate Select Algorithms
	bool disp_network;	//Plot Network using OpenGL
	bool print_network;	//Print to File
};

//CUDA Kernel Execution Parameters
struct NetworkExec {
	NetworkExec() : threads_per_block(dim3(256, 256, 1)), blocks_per_grid(dim3(256, 256, 1)) {}
	NetworkExec(dim3 tpb, dim3 bpg) : threads_per_block(tpb), blocks_per_grid(bpg) {}

	dim3 threads_per_block;
	dim3 blocks_per_grid;
};

struct NetworkProperties {
	NetworkProperties() : N(10000), k(10), dim(2), subnet_size(104857600), seed(12345L), flags(CausetFlags()), network_exec(NetworkExec()) {}
	NetworkProperties(unsigned int _N, unsigned int _k, unsigned int _dim, size_t _subnet_size, long _seed, CausetFlags _flags, NetworkExec _network_exec) : N(_N), k(_k), dim(_dim), subnet_size(_subnet_size), seed(_seed), flags(_flags), network_exec(_network_exec) {}

	CausetFlags flags;
	NetworkExec network_exec;

	unsigned int N;		//Number of Nodes
	unsigned int k;		//Average Degrees
	unsigned int dim;	//Spacetime Dimension (2 or 4)

	float a;			//Hyperboloid Pseudoradius

	size_t subnet_size;	//Maximum Contiguous Memory Request (Default 100 MB)

	long seed;
};

struct Network {
	Network() : network_properties(NetworkProperties()), nodes(NULL), links(NULL) {}
	Network(NetworkProperties _network_properties) : network_properties(_network_properties), nodes(NULL), links(NULL) {}
	Network(NetworkProperties _network_properties, Node *_nodes, unsigned int *_links) : network_properties(_network_properties), nodes(_nodes), links(_links) {}

	NetworkProperties network_properties;
	Node *nodes;
	unsigned int *links;	//Connections between Nodes

	//GPU Memory Pointers
	CUdeviceptr d_nodes;
	CUdeviceptr d_links;
};

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
	static const char *optString = ":n:k:d:s:a:h";
	static const struct option longOpts[] = {
		{ "help", no_argument, NULL, 'h' },
		{ "gpu", no_argument, NULL, 0 },
		{ "display", no_argument, NULL, 0 },
		{ "print", no_argument, NULL, 0 },
		{ "size", required_argument, NULL, 0 }
	};

	while ((c = getopt_long(argc, argv, optString, longOpts, &longIndex)) != -1) {
		switch (c) {
		case 'n':
			network_properties.N = atoi(optarg);
			break;
		case 'k':
			network_properties.k = atoi(optarg);
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
			network_properties.seed = atol(optarg);
			break;
		case 'a':
			network_properties.a = atof(optarg);
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
			break;
		case 'h':
			printf("Don't forget to make help menu\n");
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

	return network_properties;
}

#endif