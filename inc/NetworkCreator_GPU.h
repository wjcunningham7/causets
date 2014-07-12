#ifndef NETWORK_CREATOR_GPU_H_
#define NETWORK_CREATOR_GPU_H_

#include "Operations_GPU.h"
#include "Operations.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

__global__ void GenerateAdjacencyLists(float4 *nodes, int *edges, int *k_in, int *k_out, int *g_idx, int width);

__global__ void DecodeFutureEdges(uint64_t *edges, int *future_edges, int elements, int offset);

__global__ void DecodePastEdges(uint64_t *edges, int *past_edges, int elements, int offset);

__global__ void ResultingProps(int *k_in, int *k_out, int *N_res, int *N_deg2, int elements);

bool linkNodesGPU(const Node &nodes, int * const &past_edges, int * const &future_edges, int * const &past_edge_row_start, int * const &future_edge_row_start, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const double &a, const double &alpha, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &universe, const bool &verbose, const bool &bench);

#endif
