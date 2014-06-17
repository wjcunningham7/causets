#ifndef MEASUREMENTS_H_
#define MEASUREMENTS_H_

#include "Operations_GPU.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Northeastern University //
// Krioukov Research Group //
/////////////////////////////

bool measureClustering(float *& clustering, const Node * const nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start, const bool * const core_edge_exists, float &average_clustering, const int &N_tar, const int &N_deg2, const float &core_edge_fraction, Stopwatch &sMeasureClustering, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &calc_autocorr, const bool &verbose, const bool &bench);

bool measureSuccessRatio(const Node * const nodes, const int * const past_edges, const int * const future_edges, const int * const past_edge_row_start, const int * const future_edge_row_start, const bool * const core_edge_exists, float &success_ratio, const int &N_tar, const int &N_sr, const float &core_edge_fraction, Stopwatch &sMeasureSuccessRatio, const bool &verbose, const bool &bench);

bool nodesAreConnected(const Node * const nodes, const int * const future_edges, const int * const future_edge_row_start, const bool * const core_edge_exists, const int &N_tar, const float &core_edge_fraction, const int past_idx, const int future_idx);

float distance(const Node &node0, const Node &node2);

#endif
