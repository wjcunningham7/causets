/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef VALIDATE_H_
#define VALIDATE_H_

#include "Causet.h"
#include "CuResources.h"
#ifdef CUDA_ENABLED
#include "NetworkCreator_GPU.h"
#endif
#include "Operations.h"
#include "Geodesics.h"

void compareAdjacencyLists(const Node &nodes, const Edge &edges);

void compareAdjacencyListIndices(const Node &nodes, const Edge &edges);

bool compareCoreEdgeExists(const int * const k_out, const int * const future_edges, const int64_t * const future_edge_row_start, const Bitvector &adj, const int &N, const float &core_edge_fraction);

#ifdef CUDA_ENABLED
__global__ void GenerateAdjacencyLists_v1(float *w, float *x, float *y, float *z, uint64_t *edges, int *k_in, int *k_out, unsigned long long int *g_idx, int width, bool compact);

bool linkNodesGPU_v1(Node &nodes, const Edge &edges, Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const float &edge_buffer, CaResources * const ca, Stopwatch &sLinkNodesGPU, const bool &link_epso, const bool &has_exact_k, const bool &verbose, const bool &bench);

bool generateLists_v1(Node &nodes, uint64_t * const &edges, Bitvector &adj, int64_t * const &g_idx, const Spacetime &spacetime, const int &N, const double &r_max, const float &core_edge_fraction, const size_t &d_edges_size, const int &group_size, CaResources * const ca, const bool &link_epso, const bool &use_bit, const bool &verbose, const bool &bench);

bool decodeLists_v1(const Edge &edges, const uint64_t * const h_edges, const int64_t * const g_idx, const size_t &d_edges_size, CaResources * const ca, const bool &verbose);
#endif

bool printValues(Node &nodes, const Spacetime &spacetime, const int num_vals, const char *filename, const char *coord);

bool printDegrees(const Node &nodes, const int num_vals, const char *filename_in, const char *filename_out);

bool printEdgeLists(const Edge &edges, const int64_t num_vals, const char *filename_past, const char *filename_future);

bool printEdgeListPointers(const Edge &edges, const int num_vals, const char *filename_past, const char *filename_future);

bool printAdjMatrix(const Bitvector &adj, const int N, const char *filename, const int num_mpi_threads, const int rank);

bool traversePath_v1(const Node &nodes, const Edge &edges, const Bitvector &adj, bool * const &used, const Spacetime &spacetime, const int &N, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const bool &strict_routing, int source, int dest, int &nsteps, bool &success, bool &success2, bool &past_horizon);

bool measureAction_v1(uint64_t *& cardinalities, double &action, const Node &nodes, const Edge &edges, const Bitvector &adj, const Spacetime &spacetime, const int &N, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool no_pos, const bool &use_bit, const bool &verbose, const bool &bench);

bool measureAction_v2(uint64_t *& cardinalities, double &action, const Node &nodes, const Edge &edges, const Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &no_pos, const bool &use_bit, const bool &verbose, const bool &bench);

bool validateCoordinates(const Node &nodes, const Spacetime &spacetime, const double &eta0, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const int &i);

void printDot(Bitvector &adj, const int * const k_out, int N, const char *filename);

#endif
