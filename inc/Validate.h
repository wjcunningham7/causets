#ifndef VALIDATE_H_
#define VALIDATE_H_

#include "Causet.h"
#include "CuResources.h"
#ifdef CUDA_ENABLED
#include "NetworkCreator_GPU.h"
#endif
#include "Operations.h"
#include "Geodesics.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

void compareAdjacencyLists(const Node &nodes, const Edge &edges);

void compareAdjacencyListIndices(const Node &nodes, const Edge &edges);

bool compareCoreEdgeExists(const int * const k_out, const int * const future_edges, const int64_t * const future_edge_row_start, const Bitvector &adj, const int &N_tar, const float &core_edge_fraction);

#ifdef CUDA_ENABLED
__global__ void GenerateAdjacencyLists_v1(float *w, float *x, float *y, float *z, uint64_t *edges, int *k_in, int *k_out, unsigned long long int *g_idx, int width, bool compact);

bool linkNodesGPU_v1(Node &nodes, const Edge &edges, Bitvector &adj, const unsigned int &spacetime, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const float &edge_buffer, CaResources * const ca, Stopwatch &sLinkNodesGPU, const bool &verbose, const bool &bench);

bool generateLists_v1(Node &nodes, uint64_t * const &edges, Bitvector &adj, int64_t * const &g_idx, const unsigned int &spacetime, const int &N_tar, const float &core_edge_fraction, const size_t &d_edges_size, const int &group_size, CaResources * const ca, const bool &use_bit, const bool &verbose);

bool decodeLists_v1(const Edge &edges, const uint64_t * const h_edges, const int64_t * const g_idx, const size_t &d_edges_size, CaResources * const ca, const bool &verbose);
#endif

bool validateEmbedding(EVData &evd, Node &nodes, const Edge &edges, const Bitvector &adj, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const long double &N_emb, const int &N_res, const float &k_res, const double &a, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sValidateEmbedding, const bool &verbose);

bool validateDistances(DVData &dvd, Node &nodes, const unsigned int &spacetime, const int &N_tar, const double &N_dst, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sValidateDistances, const bool &verbose);

bool printValues(Node &nodes, const unsigned int &spacetime, const int num_vals, const char *filename, const char *coord);

bool printDegrees(const Node &nodes, const int num_vals, const char *filename_in, const char *filename_out);

bool printEdgeLists(const Edge &edges, const int64_t num_vals, const char *filename_past, const char *filename_future);

bool printEdgeListPointers(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future);

bool printAdjMatrix(const Bitvector &adj, const int N, const char *filename, const int num_mpi_threads, const int rank);

bool testOmega12(float tau1, float tau2, const double &omega12, const double min_lambda, const double max_lambda, const double lambda_step, const unsigned int &spacetime);

bool generateGeodesicLookupTable(const char *filename, const double max_tau, const double min_lambda, const double max_lambda, const double tau_step, const double lambda_step, const unsigned int &spacetime, const bool &verbose);

bool validateDistApprox(const Node &nodes, const Edge &edges, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha);

bool traversePath_v1(const Node &nodes, const Edge &edges, const Bitvector &adj, bool * const &used, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, int source, int dest, bool &success, bool &past_horizon);

bool measureAction_v1(int *& cardinalities, float &action, const Node &nodes, const Edge &edges, const Bitvector &adj, const unsigned int &spacetime, const int &N_tar, const int &max_cardinality, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool no_pos, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureAction_v2(int *& cardinalities, float &action, const Node &nodes, const Edge &edges, const Bitvector &adj, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const int &max_cardinality, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &no_pos, const bool &use_bit, const bool &verbose, const bool &bench);

bool validateCoordinates(const Node &nodes, const unsigned int &spacetime, const double &eta0, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const int &i);

#endif
