#ifndef NETWORK_CREATOR_GPU_H_
#define NETWORK_CREATOR_GPU_H_

#include "Operations_GPU.h"
#include "Operations.h"
#include "Validate.h"

/////////////////////////////
//(C) Will Cunningham 2014 //
// Krioukov Research Group //
// Northeastern University //
/////////////////////////////

__global__ void GenerateAdjacencyLists_v2(float *w0, float *x0, float *y0, float *z0, float *w1, float *x1, float *y1, float *z1, int *k_in, int *k_out, bool *edges, int diag, bool compact);

__global__ void GenerateAdjacencyLists_v1(float *w, float *x, float *y, float *z, uint64_t *edges, int *k_in, int *k_out, int *g_idx, int width, bool compact);

__global__ void DecodeFutureEdges(uint64_t *edges, int *future_edges, int elements, int offset);

__global__ void DecodePastEdges(uint64_t *edges, int *past_edges, int elements, int offset);

__global__ void ResultingProps(int *k_in, int *k_out, int *N_res, int *N_deg2, int elements);

bool linkNodesGPU_v2(Node &nodes, const Edge &edges, bool * const &core_edge_exists, const int &N_tar, const float &k_tar, int &N_res, float &k_res, int &N_deg2, const float &core_edge_fraction, const int &edge_buffer, Stopwatch &sLinkNodesGPU, const CUcontext &ctx, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &compact, const bool &verbose, const bool &bench);

bool generateLists_v2(Node &nodes, uint64_t * const &edges, int * const &g_idx, const int &N_tar, const size_t &d_edges_size, const CUcontext &ctx, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &compact, const bool &verbose);

bool decodeLists_v2(const Edge &edges, const uint64_t * const h_edges, const int * const g_idx, const size_t &d_edges_size, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose);

bool scanLists(const Edge &edges, const CUdeviceptr &d_k_in, const CUdeviceptr d_k_out, const int &N_tar, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose);

bool identifyListProperties(const Node &nodes, const CUdeviceptr &d_k_in, const CUdeviceptr &d_k_out, const int *g_idx, const int &N_tar, int &N_res, int &N_deg2, float &k_res, size_t &hostMemUsed, size_t &maxHostMemUsed, size_t &devMemUsed, size_t &maxDevMemUsed, const bool &verbose);

#endif
