#ifndef NETWORK_CREATOR_H_
#define NETWORK_CREATOR_H_

#include <fastmath/FastNumInt.h>

#include "Operations.h"
#ifdef CUDA_ENABLED
#include "NetworkCreator_GPU.h"
#endif
#include "Validate.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

bool initVars(NetworkProperties * const network_properties, CausetPerformance * const cp, Benchmark * const bm, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed);

bool solveExpAvgDegree(float &k_tar, double &a, double &tau0, const double &alpha, const double &delta, long &seed, const int &rank, Stopwatch &sCalcDegrees, double &bCalcDegrees, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose, const bool &bench, const int method);

bool createNetwork(Node &nodes, Edge &edges, bool *& core_edge_exists, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const float &core_edge_fraction, const int &edge_buffer, CausetMPI &cmpi, Stopwatch &sCreateNetwork, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &use_gpu, const bool &link, const bool &relink, const bool &verbose, const bool &bench, const bool &yes);

bool solveMaxTime(const int &N_tar, const float &k_tar, const int &dim, const double &a, double &zeta, double &tau0, const double &alpha, CausetMPI &cmpi, const bool &universe);

bool generateNodes(Node &nodes, const int &N_tar, const float &k_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &chi_max, const double &tau0, const double &alpha, long &seed, Stopwatch &sGenerateNodes, const bool &use_gpu, const bool &universe, const bool &compact, const bool &verbose, const bool &bench);

bool linkNodes(Node &nodes, Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &chi_max, const double &tau0, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodes, const bool &universe, const bool &compact, const bool &verbose, const bool &bench);

#endif
