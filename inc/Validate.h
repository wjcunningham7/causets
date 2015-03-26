#ifndef VALIDATE_H_
#define VALIDATE_H_

#include "Causet.h"
#include "CuResources.h"
#ifdef CUDA_ENABLED
#include "NetworkCreator_GPU.h"
#endif
#include "Operations.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

void compareAdjacencyLists(const Node &nodes, const Edge &edges);

void compareAdjacencyListIndices(const Node &nodes, const Edge &edges);

bool compareCoreEdgeExists(const int * const k_out, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction);

#ifdef CUDA_ENABLED
bool linkNodesGPU_v1(Node &nodes, const Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &compact, const bool &verbose, const bool &bench);

bool generateLists_v1(Node &nodes, uint64_t * const &edges, bool * const core_edge_exists, int * const &g_idx, const int &N_tar, const float &core_edge_fraction, const size_t &d_edges_size, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &compact, const bool &verbose);

bool decodeLists_v1(const Edge &edges, const uint64_t * const h_edges, const int * const g_idx, const size_t &d_edges_size, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose);
#endif

bool validateEmbedding(EVData &evd, Node &nodes, const Edge &edges, bool * const core_edge_exists, const int &N_tar, const float &k_tar, const double &N_emb, const int &N_res, const float &k_res, const int &dim, const Manifold &manifold, const double &a, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, long &seed, CausetMPI &cmpi, Stopwatch &sValidateEmbedding, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &compact, const bool &verbose);

bool validateDistances(DVData &dvd, Node &nodes, const int &N_tar, const double &N_dst, const int &dim, const Manifold &manifold, const double &a, const double &alpha, Stopwatch &sValidateDistances, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &compact, const bool &verbose);

bool printValues(Node &nodes, const int num_vals, const char *filename, const char *coord);

bool printDegrees(const Node &nodes, const int num_vals, const char *filename_in, const char *filename_out);

bool printEdgeLists(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future);

bool printEdgeListPointers(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future);

bool testOmega12(float tau1, float tau2, const double &omega12, const double min_lambda, const double max_lambda, const double lambda_step, const double &a, const bool &universe);

bool generateGeodesicLookupTable(const char *filename, const double max_tau, const double min_lambda, const double max_lambda, const double tau_step, const double lambda_step, const double &a, const bool &universe, const bool &verbose);

#endif
