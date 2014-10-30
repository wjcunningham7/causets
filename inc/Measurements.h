#ifndef MEASUREMENTS_H_
#define MEASUREMENTS_H_

#include "Operations.h"
#ifdef CUDA_ENABLED
#include "Operations_GPU.h"
#endif
#include "Validate.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

bool measureClustering(float *& clustering, const Node &nodes, const Edge &edges, const bool * const core_edge_exists, float &average_clustering, const int &N_tar, const int &N_deg2, const float &core_edge_fraction, Stopwatch &sMeasureClustering, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &calc_autocorr, const bool &verbose, const bool &bench);

bool measureConnectedComponents(Node &nodes, const Edge &edges, const int &N_tar, int &N_cc, int &N_gcc, Stopwatch &sMeasureConnectedComponents, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose, const bool &bench);

bool measureSuccessRatio(const Node &nodes, const Edge &edges, const bool * const core_edge_exists, float &success_ratio, const int &N_tar, const double &N_sr, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &alpha, const float &core_edge_fraction, Stopwatch &sMeasureSuccessRatio, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose, const bool &bench);

bool measureDegreeField(int *& in_degree_field, int *& out_degree_field, float &avg_idf, float &avg_odf, const float4 * const sc, const int &N_tar, int &N_df, const double &tau_m, const int &dim, const Manifold &manifold, const double &a, const double &zeta, const double &alpha, const double &delta, long &seed, Stopwatch &sMeasureDegreeField, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose, const bool &bench);

#endif
