#ifndef NETWORK_CREATOR_H_
#define NETWORK_CREATOR_H_

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include "Operations.h"
#include "GPUSubroutines.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

bool createNetwork(Node *& nodes, int *& past_edges, int *& future_edges, int *& past_edge_row_start, int *& future_edge_row_start, bool *& core_edge_exists, CUdeviceptr &d_nodes, CUdeviceptr &d_edges, const int &N_tar, const float &k_tar, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sCreateNetwork, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &use_gpu, const bool &verbose, const bool &bench, const bool &yes);
bool generateNodes(Node * const &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, long &seed, Stopwatch &sGenerateNodes, const bool &use_gpu, const bool &universe, const bool &verbose, const bool &bench);
bool linkNodes(Node * const &nodes, int * const &past_edges, int * const &future_edges, int * const &past_edge_row_start, int * const &future_edge_row_start, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodes, const bool &universe, const bool &verbose, const bool &bench);

void compareAdjacencyLists(const Node * const nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start);
void compareAdjacencyListIndices(const Node * const nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start);

bool printValues(const Node * const nodes, const int num_vals, const char *filename, const char *coord);
bool printSpatialDistances(const Node * const nodes, const Manifold &manifold, const int &N_tar, const int &dim);

#endif
