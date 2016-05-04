#ifndef NETWORK_CREATOR_H_
#define NETWORK_CREATOR_H_

#include <fastmath/FastNumInt.h>

#include "Coordinates.h"
#ifdef CUDA_ENABLED
#include "NetworkCreator_GPU.h"
#endif
#include "Validate.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

bool initVars(NetworkProperties * const network_properties, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm);

bool solveExpAvgDegree(float &k_tar, const unsigned int &spacetime, const int &N_tar, double &a, const double &r_max, double &tau0, const double &alpha, const double &delta, const int &rank, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sCalcDegrees, double &bCalcDegrees, const bool &verbose, const bool &bench, const int method);

bool createNetwork(Node &nodes, Edge &edges, Bitvector &adj, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, const int &group_size, CaResources * const ca, Stopwatch &sCreateNetwork, const bool &use_gpu, const bool &decode_cpu, const bool &link, const bool &relink, const bool &no_pos, const bool &use_bit, const bool &verbose, const bool &bench, const bool &yes);

bool generateNodes(Node &nodes, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const double &a, const double &eta0, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, CausetMPI &cmpi, MersenneRNG &mrng, Stopwatch &sGenerateNodes, const bool &verbose, const bool &bench);

bool linkNodes_v2(Node &nodes, Bitvector &adj, const unsigned int &spacetime, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, CausetMPI &cmpi, Stopwatch &sLinkNodes, const bool &use_bit, const bool &verbose, const bool &bench);

bool linkNodes(Node &nodes, Edge &edges, Bitvector &adj, const unsigned int &spacetime, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, Stopwatch &sLinkNodes, const bool &use_bit, const bool &verbose, const bool &bench);

#endif
