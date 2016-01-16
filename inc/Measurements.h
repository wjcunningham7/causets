#ifndef MEASUREMENTS_H_
#define MEASUREMENTS_H_

#include "Operations.h"
#include "Geodesics.h"
#ifdef CUDA_ENABLED
#include "Operations_GPU.h"
#include "Measurements_GPU.h"
#endif
#include "Validate.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

bool measureClustering(float *& clustering, const Node &nodes, const Edge &edges, const std::vector<bool> core_edge_exists, float &average_clustering, const int &N_tar, const int &N_deg2, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureClustering, const bool &calc_autocorr, const bool &verbose, const bool &bench);

bool measureConnectedComponents(Node &nodes, const Edge &edges, const int &N_tar, CausetMPI &cmpi, int &N_cc, int &N_gcc, CaResources * const ca, Stopwatch &sMeasureConnectedComponents, const bool &verbose, const bool &bench);

bool measureSuccessRatio(const Node &nodes, const Edge &edges, const std::vector<bool> core_edge_exists, float &success_ratio, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const double &N_sr, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureSuccessRatio, const bool &verbose, const bool &bench);

bool traversePath_v2(const Node &nodes, const Edge &edges, const std::vector<bool> core_edge_exists, bool * const &used, const double * const table, const unsigned int &spacetime, const int &N_tar, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const long &size, int source, int dest, bool &success);

bool measureDegreeField(int *& in_degree_field, int *& out_degree_field, float &avg_idf, float &avg_odf, Coordinates *& c, const unsigned int &spacetime, const int &N_tar, int &N_df, const double &tau_m, const double &a, const double &zeta, const double &zeta1, const double &alpha, const double &delta, CaResources * const ca, Stopwatch &sMeasureDegreeField, const bool &verbose, const bool &bench);

bool measureAction_v2(int *& cardinalities, float &action, const Node &nodes, const Edge &edges, const std::vector<bool> core_edge_exists, const unsigned int &spacetime, const int &N_tar, const float &k_tar, const int &max_cardinality, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &no_pos, const bool &use_bit, const bool &verbose, const bool &bench);

#endif
