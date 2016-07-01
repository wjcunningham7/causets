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

bool measureSuccessRatio(const Node &nodes, const Edge &edges, const Bitvector &adj, float &success_ratio, float &success_ratio2, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const long double &N_sr, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureSuccessRatio, const bool &use_bit, const bool &verbose, const bool &bench);

bool traversePath_v2(const Node &nodes, const Edge &edges, const Bitvector &adj, bool * const &used, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, int source, int dest, bool &success);

bool traversePath_v3(const Node &nodes, const Bitvector &adj, bool * const &used, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, int source, int dest, bool &success);

bool measureDegreeField(int *& in_degree_field, int *& out_degree_field, float &avg_idf, float &avg_odf, Coordinates *& c, const unsigned int &spacetime, const int &N_tar, int &N_df, const double &tau_m, const double &a, const double &zeta, const double &zeta1, const double &alpha, const double &delta, CaResources * const ca, Stopwatch &sMeasureDegreeField, const bool &verbose, const bool &bench);

void* actionKernel(void *params);

bool measureAction_v5(uint64_t *& cardinalities, float &action, Bitvector &adj, const unsigned int &spacetime, const int &N_tar, CausetMPI &cmpi, CaResources * const ca, Stopwatch sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureAction_v4(uint64_t *& cardinalities, float &action, Bitvector &adj, const unsigned int &spacetime, const int &N_tar, CausetMPI &cmpi, CaResources * const ca, Stopwatch sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureAction_v3(uint64_t *& cardinalities, float &action, Bitvector &adj, const unsigned int &spacetime, const int &N_tar, CaResources * const ca, Stopwatch sMeasureAction, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureVecprod(float *& vecprods, const Node &nodes, const int spacetime, const int N_tar, const long double N_vp, const double a, const double zeta, const double r_max, const double tau0, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureVecprod, const bool verbose, const bool bench);

#endif
