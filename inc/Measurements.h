/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef MEASUREMENTS_H_
#define MEASUREMENTS_H_

#include "Operations.h"
#include "Geodesics.h"
#include "SMI.h"
#ifdef CUDA_ENABLED
#include "Operations_GPU.h"
#endif
#include "Validate.h"

//#include <cuComplex.h>
//#include <cusolverDn.h>
struct action_params {
	Bitvector *adj;
	std::vector<unsigned int> *current;

	uint64_t *cardinalities;
	uint64_t npairs;
	int N;
	int N_eff;

	int rank;
	int num_mpi_threads;

	bool *busy;
};

bool measureClustering(float *& clustering, const Node &nodes, const Edge &edges, const Bitvector &adj, float &average_clustering, const int &N, const int &N_deg2, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureClustering, const bool &verbose, const bool &bench);

bool measureConnectedComponents(Node &nodes, const Edge &edges, const Bitvector &adj, const int &N, CausetMPI &cmpi, int &N_cc, int &N_gcc, CaResources * const ca, Stopwatch &sMeasureConnectedComponents, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureSuccessRatio(const Node &nodes, const Edge &edges, const Bitvector &adj, float &success_ratio, float &success_ratio2, float &stretch, const Spacetime &spacetime, const int &N, const float &k_tar, const long double &N_sr, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureSuccessRatio, const bool &link_epso, const bool &use_bit, const bool &calc_stretch, const bool &strict_routing, const bool &verbose, const bool &bench);

bool traversePath_v2(const Node &nodes, const Edge &edges, const Bitvector &adj, bool * const &used, const Spacetime &spacetime, const int &N, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const bool &strict_routing, int source, int dest, bool &success);

bool traversePath_v3(const Node &nodes, const Bitvector &adj, bool * const &used, const Spacetime &spacetime, const int &N, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const bool &link_epso, const bool &strict_routing, int source, int dest, bool &success);

void* actionKernel(void *params);

bool measureAction_v6(uint64_t *& cardinalities, double &action, Bitvector &adj, const Spacetime &spacetime, const int N, const double epsilon, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureAction, const bool use_bit, const bool split_mpi, const bool verbose, const bool bench);

bool measureAction_v5(uint64_t *& cardinalities, double &action, Bitvector &adj, const Spacetime &spacetime, const int &N, const double epsilon, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureAction_v4(uint64_t *& cardinalities, double &action, Bitvector &adj, const Spacetime &spacetime, const int &N, const double epsilon, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureAction_v3(uint64_t *& cardinalities, double &action, Bitvector &adj, const int * const k_in, const int * const k_out, const Spacetime &spacetime, const int N, const double epsilon, const GraphType gt, CaResources * const ca, Stopwatch &sMeasureAction, const bool use_bit, const bool verbose, const bool bench);

bool timelikeActionCandidates(std::vector<unsigned int> &candidates, int *chaintime, const Node &nodes, Bitvector &adj, const int * const k_in, const int * const k_out, const Spacetime &spacetime, const int &N, CaResources * const ca, Stopwatch sMeasureActionTimelike, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureTimelikeAction(Network * const graph, Network * const subgraph, const std::vector<unsigned int> &candidates, CaResources * const ca);

bool measureTheoreticalAction(double *& actth, int N_actth, const Node &nodes, Bitvector &adj, const Spacetime &spacetime, const int N, const double eta0, const double delta, CaResources * const ca, Stopwatch &sMeasureThAction, const bool verbose, const bool bench);

bool measureChain(Bitvector &longest_chains, int &chain_length, std::pair<int,int> &longest_pair, const Node &nodes, Bitvector &adj, const int N, CaResources * const ca, Stopwatch &sMeasureChain, const bool verbose, const bool bench);

bool measureHubDensity(float &hub_density, float *& hub_densities, Bitvector &adj, const int * const k_in, const int * const k_out, const int N, int N_hubs, CaResources * const ca, Stopwatch &sMeasureHubs, const bool verbose, const bool bench);

bool measureGeoDis(float &geo_discon, const Node &nodes, const Edge &edges, const Bitvector &adj, const Spacetime &spacetime, const int N, const long double N_gd, const double a, const double zeta, const double zeta1, const double r_max, const double alpha, const float core_edge_fraction, MersenneRNG &mrng, Stopwatch &sMeasureGeoDis, const bool use_bit, const bool verbose, const bool bench);

bool measureFoliation(Bitvector &timelike_foliation, Bitvector &spacelike_foliation, std::vector<unsigned int> &ax_set, Bitvector &adj, const Node &nodes, const int N, CaResources * const ca, Stopwatch &sMeasureFoliation, const bool verbose, const bool bench);

bool measureAntichain(const Bitvector &adj, const Node &nodes, const int N, MersenneRNG &mrng, Stopwatch &sMeasureAntichain, const bool verbose, const bool bench);

bool measureDimension(float &dimension, Bitvector &adj, const Spacetime &spacetime, const int N, std::pair<int,int> longest_pair, Stopwatch &sMeasureDimension, const bool verbose, const bool bench);

bool measureSpacetimeMutualInformation(uint64_t *& smi, Bitvector &adj, const Node &nodes, const Spacetime &spacetime, const int N, const double eta0, const double r_max, CaResources * const ca, Stopwatch &sMeasureSMI, const bool verbose, const bool bench);

bool measureExtrinsicCurvature(Network *network, CaResources * const ca, CUcontext ctx);

//bool measureEntanglementEntropy(const Bitvector &adj, const Spacetime &spacetime, const int N, Stopwatch &sMeasureEntanglementEntropy, const bool verbose, const bool bench);

#endif
