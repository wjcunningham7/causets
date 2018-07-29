/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef NETWORK_CREATOR_H_
#define NETWORK_CREATOR_H_

#include <FastNumInt.h>

#include "Coordinates.h"
#ifdef CUDA_ENABLED
#include "NetworkCreator_GPU.h"
#endif
#include "Validate.h"

bool initVars(NetworkProperties * const network_properties, CaResources * const ca, CausetPerformance * const cp, Benchmark * const bm);

bool solveExpAvgDegree(float &k_tar, const Spacetime &spacetime, const int &N, double &a, const double &r_max, double &tau0, const double &alpha, const double &delta, const int &rank, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sCalcDegrees, double &bCalcDegrees, const bool &verbose, const bool &bench, const int method);

bool createNetwork(Node &nodes, Edge &edges, Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, const float &core_edge_fraction, const float &edge_buffer, const GraphType &gt, CausetMPI &cmpi, const int &group_size, CaResources * const ca, Stopwatch &sCreateNetwork, const bool &use_gpu, const bool &decode_cpu, const bool &link, const bool &relink, const bool &no_pos, const bool &use_bit, const bool &mpi_split, const bool &verbose, const bool &bench, const bool &yes);

bool generateNodes(Node &nodes, const Spacetime &spacetime, const int &N, const float &k_tar, const double &a, const double &eta0, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, const double gamma, CausetMPI &cmpi, MersenneRNG &mrng, Stopwatch &sGenerateNodes, const bool &growing, const bool &verbose, const bool &bench);

bool generateKROrder(Node &nodes, Bitvector &adj, const int N_tar, const int N, int &N_res, float &k_res, int &N_deg2, Stopwatch &sGenKR, const bool verbose, const bool bench);

bool generateRandomOrder(Node &nodes, Bitvector &adj, const int N, int &N_res, float &k_res, int &N_deg2, MersenneRNG &mrng, Stopwatch &sGenRandom, const bool verbose, const bool bench);

bool linkNodes_v2(Node &nodes, Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, const double gamma, CausetMPI &cmpi, Stopwatch &sLinkNodes, const bool &link_epso, const bool &has_exact_k, const bool &use_bit, const bool &mpi_split, const bool &verbose, const bool &bench);

bool linkNodes_v1(Node &nodes, Edge &edges, Bitvector &adj, const Spacetime &spacetime, const int &N, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &tau0, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, Stopwatch &sLinkNodes, const bool &link_epso, const bool &has_exact_k, const bool &use_bit, const bool &verbose, const bool &bench);

#endif
