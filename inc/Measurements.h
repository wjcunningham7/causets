#ifndef MEASUREMENTS_H_
#define MEASUREMENTS_H_

#include "Operations.h"
#include "Geodesics.h"
#ifdef CUDA_ENABLED
#include "Operations_GPU.h"
#endif
#include "Validate.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

struct action_params {
	Bitvector *adj;
	Bitvector *workspace;
	std::vector<unsigned int> *current;

	uint64_t *cardinalities;
	uint64_t clone_length;
	uint64_t npairs;
	int N_tar;
	int N_eff;

	int rank;
	int num_mpi_threads;

	bool *busy;
};

bool measureClustering(float *& clustering, const Node &nodes, const Edge &edges, const Bitvector &adj, float &average_clustering, const int &N_tar, const int &N_deg2, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureClustering, const bool &calc_autocorr, const bool &verbose, const bool &bench);

bool measureConnectedComponents(Node &nodes, const Edge &edges, const Bitvector &adj, const int &N_tar, CausetMPI &cmpi, int &N_cc, int &N_gcc, CaResources * const ca, Stopwatch &sMeasureConnectedComponents, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureSuccessRatio(const Node &nodes, const Edge &edges, const Bitvector &adj, float &success_ratio, float &success_ratio2, float &stretch, const Spacetime &spacetime, const int &N_tar, const float &k_tar, const long double &N_sr, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureSuccessRatio, const bool &link_epso, const bool &use_bit, const bool &calc_stretch, const bool &strict_routing, const bool &verbose, const bool &bench);

bool traversePath_v2(const Node &nodes, const Edge &edges, const Bitvector &adj, bool * const &used, const Spacetime &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const bool &strict_routing, int source, int dest, bool &success);

bool traversePath_v3(const Node &nodes, const Bitvector &adj, bool * const &used, const Spacetime &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const bool &link_epso, const bool &strict_routing, int source, int dest, bool &success);

bool measureDegreeField(int *& in_degree_field, int *& out_degree_field, float &avg_idf, float &avg_odf, Coordinates *& c, const Spacetime &spacetime, const int &N_tar, int &N_df, const double &tau_m, const double &a, const double &zeta, const double &zeta1, const double &alpha, const double &delta, CaResources * const ca, Stopwatch &sMeasureDegreeField, const bool &verbose, const bool &bench);

void* actionKernel(void *params);

bool measureAction_v6(uint64_t *& cardinalities, float &action, Bitvector &adj, const Spacetime &spacetime, const int N_tar, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureAction, const bool use_bit, const bool split_mpi, const bool verbose, const bool bench);

bool measureAction_v5(uint64_t *& cardinalities, float &action, Bitvector &adj, const Spacetime &spacetime, const int &N_tar, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureAction_v4(uint64_t *& cardinalities, float &action, Bitvector &adj, const Spacetime &spacetime, const int &N_tar, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureAction_v3(uint64_t *& cardinalities, float &action, Bitvector &adj, const int * const k_in, const int * const k_out, const Spacetime &spacetime, const int &N_tar, CaResources * const ca, Stopwatch &sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench);

bool timelikeActionCandidates(std::vector<unsigned int> &candidates, int *chaintime, const Node &nodes, Bitvector &adj, const int * const k_in, const int * const k_out, const Spacetime &spacetime, const int &N_tar, CaResources * const ca, Stopwatch sMeasureActionTimelike, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureTimelikeAction(Network * const graph, Network * const subgraph, const std::vector<unsigned int> &candidates, CaResources * const ca);

bool measureTheoreticalAction(double *& actth, int N_actth, const Node &nodes, Bitvector &adj, const Spacetime &spacetime, const int N_tar, const double eta0, const double delta, CaResources * const ca, Stopwatch &sMeasureThAction, const bool verbose, const bool bench);

bool measureVecprod(float *& vecprods, const Node &nodes, const Spacetime &spacetime, const int N_tar, const long double N_vp, const double a, const double zeta, const double r_max, const double tau0, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureVecprod, const bool verbose, const bool bench);

bool measureChain(int &chain_sym, int &chain_asym, Bitvector &adj, Bitvector &subadj, const Spacetime &spacetime, const int N, const int N_sub, CaResources * const ca, Stopwatch &sMeasureChain, const bool verbose, const bool bench);

bool measureHubDensity(float &hub_density, float *& hub_densities, Bitvector &adj, const int * const k_in, const int * const k_out, const int N_tar, int N_hubs, CaResources * const ca, Stopwatch &sMeasureHubs, const bool verbose, const bool bench);

bool measureGeoDis(float &geo_discon, const Node &nodes, const Edge &edges, const Bitvector &adj, const Spacetime &spacetime, const int N_tar, const long double N_gd, const double a, const double zeta, const double zeta1, const double r_max, const double alpha, const float core_edge_fraction, MersenneRNG &mrng, Stopwatch &sMeasureGeoDis, const bool use_bit, const bool verbose, const bool bench);

#endif
