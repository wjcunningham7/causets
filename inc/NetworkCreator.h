#ifndef NETWORK_CREATOR_H_
#define NETWORK_CREATOR_H_

#include <fastmath/FastNumInt.h>

#include "Operations.h"
#include "NetworkCreator_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

bool initVars(NetworkProperties * const network_properties);

bool createNetwork(Node &nodes, Edge &edges, bool *& core_edge_exists, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sCreateNetwork, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &use_gpu, const bool &verbose, const bool &bench, const bool &yes);

bool solveMaxTime(const int &N_tar, const float &k_tar, const int &dim, const double &a, double &zeta, double &tau0, const double &alpha, const bool &universe);

bool generateNodes(const Node &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, long &seed, Stopwatch &sGenerateNodes, const bool &use_gpu, const bool &universe, const bool &verbose, const bool &bench);

bool linkNodes(const Node &nodes, Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &tau0, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodes, const bool &universe, const bool &verbose, const bool &bench);

void compareAdjacencyLists(const Node &nodes, const Edge &edges);

void compareAdjacencyListIndices(const Node &nodes, const Edge &edges);

bool printValues(const Node &nodes, const int num_vals, const char *filename, const char *coord);

#endif
