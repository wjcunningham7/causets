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
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

void compareAdjacencyLists(const Node &nodes, const Edge &edges);

void compareAdjacencyListIndices(const Node &nodes, const Edge &edges);

bool compareCoreEdgeExists(const int * const k_out, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction);

#ifdef CUDA_ENABLED
__global__ void GenerateAdjacencyLists_v1(float *w, float *x, float *y, float *z, uint64_t *edges, int *k_in, int *k_out, int *g_idx, int width, bool compact);

bool linkNodesGPU_v1(Node &nodes, const Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const float &edge_buffer, CaResources * const ca, Stopwatch &sLinkNodesGPU, const bool &compact, const bool &verbose, const bool &bench);

bool generateLists_v1(Node &nodes, uint64_t * const &edges, bool * const core_edge_exists, int * const &g_idx, const int &N_tar, const float &core_edge_fraction, const size_t &d_edges_size, CaResources * const ca, const bool &compact, const bool &verbose);

bool decodeLists_v1(const Edge &edges, const uint64_t * const h_edges, const int * const g_idx, const size_t &d_edges_size, CaResources * const ca, const bool &verbose);
#endif

bool validateEmbedding(EVData &evd, Node &nodes, const Edge &edges, bool * const core_edge_exists, const int &N_tar, const float &k_tar, const double &N_emb, const int &N_res, const float &k_res, const int &dim, const Manifold &manifold, const double &a, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, long &seed, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sValidateEmbedding, const bool &compact, const bool &verbose);

bool validateDistances(DVData &dvd, Node &nodes, const int &N_tar, const double &N_dst, const int &dim, const Manifold &manifold, const double &a, const double &alpha, long &seed, CaResources * const ca, Stopwatch &sValidateDistances, const bool &compact, const bool &verbose);

bool printValues(Node &nodes, const int num_vals, const char *filename, const char *coord);

bool printDegrees(const Node &nodes, const int num_vals, const char *filename_in, const char *filename_out);

bool printEdgeLists(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future);

bool printEdgeListPointers(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future);

bool testOmega12(float tau1, float tau2, const double &omega12, const double min_lambda, const double max_lambda, const double lambda_step, const Manifold &manifold);

bool generateGeodesicLookupTable(const char *filename, const double max_tau, const double min_lambda, const double max_lambda, const double tau_step, const double lambda_step, const Manifold &manifold, const bool &verbose);

bool validateDistApprox(const Node &nodes, const Edge &edges, const int &N_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &chi_max, const double &alpha, const bool &compact);

bool traversePath_v1(const Node &nodes, const Edge &edges, const bool * const core_edge_exists, bool * const &used, const double * const table, const int &N_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &chi_max, const double &alpha, const float &core_edge_fraction, const long &size, const bool &compact, int source, int dest, bool &success);

bool measureAction_v1(int *& cardinalities, float &action, const Node &nodes, const Edge &edges, const bool * const core_edge_exists, const int &N_tar, const int &max_cardinality, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &chi_max, const double &alpha, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &compact, const bool &verbose, const bool &bench);

#endif
