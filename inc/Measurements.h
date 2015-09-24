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
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

bool measureClustering(float *& clustering, const Node &nodes, const Edge &edges, const bool * const core_edge_exists, float &average_clustering, const int &N_tar, const int &N_deg2, const float &core_edge_fraction, CaResources * const ca, Stopwatch &sMeasureClustering, const bool &calc_autocorr, const bool &verbose, const bool &bench);

bool measureConnectedComponents(Node &nodes, const Edge &edges, const int &N_tar, CausetMPI &cmpi, int &N_cc, int &N_gcc, CaResources * const ca, Stopwatch &sMeasureConnectedComponents, const bool &verbose, const bool &bench);

bool measureSuccessRatio(const Node &nodes, const Edge &edges, bool * const core_edge_exists, float &success_ratio, const int &N_tar, const float &k_tar, const double &N_sr, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, MersenneRNG &mrng, CaResources * const ca, Stopwatch &sMeasureSuccessRatio, const bool &compact, const bool &verbose, const bool &bench);

bool traversePath_v2(const Node &nodes, const Edge &edges, const bool * const core_edge_exists, bool * const &used, const double * const table, const int &N_tar, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const long &size, const bool &compact, int source, int dest, bool &success);

bool measureDegreeField(int *& in_degree_field, int *& out_degree_field, float &avg_idf, float &avg_odf, Coordinates *& c, const int &N_tar, int &N_df, const double &tau_m, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &alpha, const double &delta, CaResources * const ca, Stopwatch &sMeasureDegreeField, const bool &compact, const bool &verbose, const bool &bench);

bool measureAction_v2(int *& cardinalities, float &action, const Node &nodes, const Edge &edges, bool * const core_edge_exists, const int &N_tar, const float &k_tar, const int &max_cardinality, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &zeta1, const double &r_max, const double &alpha, const float &core_edge_fraction, const float &edge_buffer, CausetMPI &cmpi, CaResources * const ca, Stopwatch &sMeasureAction, const bool &link, const bool &relink, const bool &compact, const bool &verbose, const bool &bench);

#endif
